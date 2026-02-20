import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, str(default))).strip())
    except Exception:
        return int(default)


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


class _LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear.

    Computes: y = base(x) + scale * ( (dropout(x) @ A^T) @ B^T )
      - A: [r, in_features], B: [out_features, r]
      - base weights stay frozen; LoRA params are trainable.
    """

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA expects nn.Linear, got {type(base)}")
        self.base = base
        self.r = int(r)
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be >0, got r={r}")

        in_features = int(base.in_features)
        out_features = int(base.out_features)

        self.lora_A = nn.Parameter(torch.empty((self.r, in_features), dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty((out_features, self.r), dtype=torch.float32))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B)

        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(self.r) if float(alpha) > 0 else (1.0 / float(self.r))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        # Freeze base weights explicitly (caller may freeze the full model too).
        try:
            self.base.weight.requires_grad = False
        except Exception:
            pass
        try:
            if self.base.bias is not None:
                self.base.bias.requires_grad = False
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x2 = self.dropout(x)
        # Keep LoRA math in fp32 for stability; cast delta back to base output dtype.
        x2 = x2.to(dtype=torch.float32)
        delta = torch.matmul(x2, self.lora_A.t())
        delta = torch.matmul(delta, self.lora_B.t()) * self.scaling
        return y + delta.to(dtype=y.dtype)


def _maybe_enable_llama_lora(model: nn.Module) -> int:
    """
    Enable a small LoRA adapter on Llama self-attn projections.

    Controlled via env:
      - VIGOR_LLM_LORA=1
      - VIGOR_LLM_LORA_R (default 8)
      - VIGOR_LLM_LORA_ALPHA (default 16)
      - VIGOR_LLM_LORA_DROPOUT (default 0.0)
      - VIGOR_LLM_LORA_LAST_N (default 4; apply to last N layers; 0 => all)
      - VIGOR_LLM_LORA_TARGETS (default "q_proj,v_proj")

    Returns: number of wrapped Linear modules.
    """
    if not _env_flag("VIGOR_LLM_LORA", "0"):
        return 0

    r = _env_int("VIGOR_LLM_LORA_R", 8)
    alpha = _env_float("VIGOR_LLM_LORA_ALPHA", 16.0)
    dropout = _env_float("VIGOR_LLM_LORA_DROPOUT", 0.0)
    last_n = _env_int("VIGOR_LLM_LORA_LAST_N", 4)
    targets = [t.strip() for t in _env_str("VIGOR_LLM_LORA_TARGETS", "q_proj,v_proj").split(",") if t.strip()]
    if not targets:
        targets = ["q_proj", "v_proj"]

    # LlamaForCausalLM -> `.model` is the decoder-only base; `.model.layers` holds blocks.
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None:
        # Fallback: try `model.model.layers` (some wrappers nest one more `.model`).
        base2 = getattr(base, "model", None) if base is not None else None
        layers = getattr(base2, "layers", None)
        base = base2 if layers is not None else base
    if layers is None:
        raise RuntimeError("VIGOR_LLM_LORA=1 but cannot locate Llama layers (expected model.model.layers).")

    num_layers = int(len(layers))
    if last_n <= 0:
        start = 0
    else:
        start = max(0, num_layers - int(last_n))

    wrapped = 0
    for i in range(start, num_layers):
        layer = layers[i]
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for name in targets:
            if not hasattr(attn, name):
                continue
            m = getattr(attn, name)
            if isinstance(m, _LoRALinear):
                continue
            if not isinstance(m, nn.Linear):
                continue
            setattr(attn, name, _LoRALinear(m, r=r, alpha=alpha, dropout=dropout))
            wrapped += 1

    print(
        f"[Vigor][llama_stepslot][lora] enabled=1 r={r} alpha={alpha} dropout={dropout} "
        f"last_n={last_n} targets={targets} wrapped={wrapped}",
        flush=True,
    )
    return int(wrapped)


def _safe_get_referential_token(ref_order, sample_idx: int, step_idx: int) -> str:
    """
    Vigor DataLoader may collate `referential_order` in either layout:
      - list[batch] of list[order_len] strings, or
      - list[order_len] of list[batch] strings.
    """
    if ref_order is None:
        return ""
    try:
        return ref_order[sample_idx][step_idx]
    except Exception:
        try:
            return ref_order[step_idx][sample_idx]
        except Exception:
            return ""


def _maybe_build_order_valid_mask(batch: dict, order_len: int, device: torch.device) -> Optional[torch.Tensor]:
    """
    Build a per-sample validity mask from `batch["ori_order_len"]`:
      mask[b,k] = 1 iff k < ori_order_len[b]
    Shape: [B, order_len] float32 on `device`.
    """
    if not isinstance(batch, dict):
        return None
    ori_len = batch.get("ori_order_len", None)
    if ori_len is None:
        return None
    try:
        ori_len_t = torch.as_tensor(ori_len, device=device).long().view(-1)
    except Exception:
        return None
    if ori_len_t.numel() == 0:
        return None
    steps = torch.arange(int(order_len), device=device).view(1, -1)
    return (steps < ori_len_t.view(-1, 1)).to(dtype=torch.float32)

def _predict_order_valid_mask_from_stop(
    order_embeds: torch.Tensor,
    stop_embed: torch.Tensor,
    *,
    tau: float,
) -> torch.Tensor:
    """
    Predict a binary validity mask m_k from STOP similarity.

    Rule: find the first step whose cosine(order_embeds_k, stop_embed) >= tau.
          All steps before it are valid; from that step onward invalid.
          If no step crosses tau, keep all steps valid.

    Returns: float mask of shape [B, O] on the same device as `order_embeds`.
    """
    if (not torch.is_tensor(order_embeds)) or order_embeds.dim() != 3:
        raise RuntimeError("order_embeds must be a tensor [B,O,D]")
    if not torch.is_tensor(stop_embed):
        raise RuntimeError("stop_embed must be a tensor [D]")
    B, O, _ = order_embeds.shape
    tau_f = float(tau)
    stop_n = F.normalize(stop_embed.detach().float().view(1, 1, -1), dim=-1)
    emb_n = F.normalize(order_embeds.detach().float(), dim=-1)
    cos = (emb_n * stop_n).sum(dim=-1)  # [B,O]
    stop_hit = cos >= tau_f
    any_hit = stop_hit.any(dim=1)
    first_stop = torch.where(any_hit, stop_hit.float().argmax(dim=1), torch.full((B,), O, device=cos.device))
    idx = torch.arange(O, device=cos.device).view(1, -1).expand(B, -1)
    pred = (idx < first_stop.view(-1, 1)).to(dtype=torch.float32)
    return pred

def _predict_order_valid_mask_changepoint_from_stop(
    order_embeds: torch.Tensor,
    stop_embed: torch.Tensor,
    *,
    min_score: float = 0.05,
) -> torch.Tensor:
    """
    Predict a binary validity mask m_k from STOP similarity using a changepoint heuristic.

    Instead of a fixed threshold tau, we choose a split point L (1..O) per sample by maximizing
    separation between "valid" and "invalid" cosine similarities:

      score(L) = mean(cos[L:]) - mean(cos[:L])   for L in {1..O-1}

    If the best score is below `min_score`, we fall back to L=O (keep all steps valid).
    """
    if (not torch.is_tensor(order_embeds)) or order_embeds.dim() != 3:
        raise RuntimeError("order_embeds must be a tensor [B,O,D]")
    if not torch.is_tensor(stop_embed):
        raise RuntimeError("stop_embed must be a tensor [D]")
    B, O, _ = order_embeds.shape
    if O <= 0:
        raise RuntimeError("order_embeds has empty order dimension")
    if O == 1:
        return torch.ones((B, 1), device=order_embeds.device, dtype=torch.float32)

    stop_n = F.normalize(stop_embed.detach().float().view(1, 1, -1), dim=-1)
    emb_n = F.normalize(order_embeds.detach().float(), dim=-1)
    cos = (emb_n * stop_n).sum(dim=-1)  # [B,O]

    total = cos.sum(dim=1)  # [B]
    csum = cos.cumsum(dim=1)  # [B,O]

    # Candidate split L corresponds to a boundary after step L-1 (0-based).
    # Evaluate L in {1..O-1}. (L=O means "no stop", handled by fallback.)
    scores = []
    for L in range(1, int(O)):
        valid_sum = csum[:, L - 1]
        invalid_sum = total - valid_sum
        valid_mean = valid_sum / float(L)
        invalid_mean = invalid_sum / float(O - L)
        scores.append((invalid_mean - valid_mean).unsqueeze(1))
    score_mat = torch.cat(scores, dim=1)  # [B,O-1]

    best_score, best_idx = score_mat.max(dim=1)  # best_idx in [0..O-2] => L=best_idx+1
    L_hat = best_idx + 1
    # If separation is too small, treat as "no stop" => L=O.
    L_hat = torch.where(best_score >= float(min_score), L_hat, torch.full_like(L_hat, int(O)))

    idx = torch.arange(int(O), device=order_embeds.device).view(1, -1).expand(B, -1)
    pred = (idx < L_hat.view(-1, 1)).to(dtype=torch.float32)
    return pred


def _predict_order_valid_mask_from_gate_probs(
    stop_prob: torch.Tensor,
    *,
    tau: float,
    mode: str = "threshold",
    monotonic: bool = False,
) -> torch.Tensor:
    """
    Predict validity mask m_k from a per-step STOP probability.

    Input:
      stop_prob: [B,O] with values in [0,1]
    Rule:
      first_stop = first k where stop_prob[b,k] >= tau (else O)
      valid mask: k < first_stop
    """
    if (not torch.is_tensor(stop_prob)) or stop_prob.dim() != 2:
        raise RuntimeError("stop_prob must be a tensor [B,O]")
    B, O = stop_prob.shape
    mode = str(mode or "threshold").strip().lower()
    p = stop_prob
    if monotonic and O > 1:
        # Enforce non-decreasing STOP probability along the chain.
        p = torch.cummax(p, dim=1).values

    if mode in {"map", "argmax", "nll"}:
        # MAP decode under an independence + prefix/suffix factorization:
        #   score(L) = sum_{k<L} log(1-p_k) + sum_{k>=L} log(p_k),  L in [0..O]
        eps = 1e-6
        logp = (p.clamp(min=eps, max=1.0 - eps)).log()
        log1p = (1.0 - p).clamp(min=eps, max=1.0 - eps).log()

        # prefix_sum[:, L] = sum_{k < L} log(1-p_k)
        prefix = torch.zeros((B, O + 1), device=p.device, dtype=p.dtype)
        prefix[:, 1:] = log1p.cumsum(dim=1)
        # suffix_sum[:, L] = sum_{k >= L} log(p_k)
        suffix = torch.zeros((B, O + 1), device=p.device, dtype=p.dtype)
        suffix[:, :O] = logp.flip(1).cumsum(1).flip(1)

        scores = prefix + suffix
        first_stop = scores.argmax(dim=1)  # [B] in [0..O]
    else:
        # Threshold decode: first index with p_k >= tau (else O).
        tau_f = float(tau)
        stop_hit = p >= tau_f
        any_hit = stop_hit.any(dim=1)
        first_stop = torch.where(any_hit, stop_hit.float().argmax(dim=1), torch.full((B,), O, device=p.device))

    idx = torch.arange(O, device=stop_prob.device).view(1, -1).expand(B, -1)
    pred = (idx < first_stop.view(-1, 1)).to(dtype=torch.float32)
    return pred


@dataclass
class LlamaStepSlotConfig:
    model_path: str
    order_len: int = 4
    max_length: int = 64
    memory_tokens: int = 0  # soft memory tokens appended via inputs_embeds (no vocab expansion)
    distill_w: float = 1.0
    distill_type: str = "cos"  # "cos" | "mse"
    global_distill_w: float = 1.0
    global_distill_type: str = "cos"  # "cos" | "mse"
    freeze_llm_except_step_rows: bool = True
    local_files_only: bool = True
    use_bf16: bool = True


class LlamaStepSlotOrderEncoder(nn.Module):
    """
    Causal-LM step-slot encoder for Mask3D-Vigor:
    - Per step k, build an isolated text: "{utterance} {step_text} <step{k}>"
      and take the hidden state at the <step{k}> position (last non-pad token).
    - Output is projected to Vigor's `inner_dim` (typically 768).

    This avoids "prefix leakage" across steps because each step is encoded by a
    separate LM forward pass (batched as B*order_len).

    Experimental alternative (enable via env `VIGOR_LLM_STEPSLOT_ONEPASS=1` in the wrapper):
    - Build ONE causal-LM sequence per sample that contains the utterance + all steps,
      and read the hidden states at each <stepK> position as `order_embeds`.
    - If `memory_tokens>0`, we append soft memory tokens in the same forward pass and use
      them as `lang_embeds`, so the entire LLM side is computed with a single forward().
    """

    def __init__(self, out_dim: int, cfg: LlamaStepSlotConfig):
        super().__init__()
        self.cfg = cfg
        self.out_dim = int(out_dim)
        self.order_len = int(cfg.order_len)
        self.n_mem = int(getattr(cfg, "memory_tokens", 0) or 0)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError("transformers is required for LlamaStepSlotOrderEncoder") from e

        torch_dtype = None
        if cfg.use_bf16:
            torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path,
            use_fast=True,
            local_files_only=bool(cfg.local_files_only),
            trust_remote_code=True,
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch_dtype,
            local_files_only=bool(cfg.local_files_only),
            trust_remote_code=True,
        )

        self.model.config.use_cache = False
        self.model.eval()

        step_tokens = [f"<step{i+1}>" for i in range(self.order_len)]
        global_token = "<cls>"
        stop_token = str(os.environ.get("VIGOR_STOP_TOKEN", "<STOP>")).strip() or "<STOP>"
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": step_tokens})
        added += self.tokenizer.add_special_tokens({"additional_special_tokens": [global_token]})
        added += self.tokenizer.add_special_tokens({"additional_special_tokens": [stop_token]})
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.step_token_ids = [int(self.tokenizer.convert_tokens_to_ids(t)) for t in step_tokens]
        self.global_token = global_token
        self.global_token_id = int(self.tokenizer.convert_tokens_to_ids(global_token))
        self.stop_token = stop_token
        self.stop_token_id = int(self.tokenizer.convert_tokens_to_ids(stop_token))

        # Optional: attach small LoRA adapters to help one-pass learn "routing/separation"
        # without updating the base LLM weights.
        self._lora_wrapped = 0
        try:
            self._lora_wrapped = _maybe_enable_llama_lora(self.model)
        except Exception as e:
            if _env_flag("VIGOR_LLM_LORA", "0"):
                raise
            else:
                print(f"[Vigor][llama_stepslot][lora][warn] {type(e).__name__}: {e}", flush=True)

        self.proj_step = nn.Sequential(
            nn.Linear(int(self._hidden_size()), self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.proj_global = nn.Sequential(
            nn.Linear(int(self._hidden_size()), self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        self.proj_mem = None
        if self.n_mem > 0:
            self.mem_tokens = nn.Parameter(torch.empty(self.n_mem, int(self._hidden_size())))
            nn.init.normal_(self.mem_tokens, mean=0.0, std=0.02)
            self.proj_mem = nn.Sequential(
                nn.Linear(int(self._hidden_size()), self.out_dim),
                nn.LayerNorm(self.out_dim),
            )
        else:
            self.mem_tokens = None

        if cfg.freeze_llm_except_step_rows:
            for p in self.model.parameters():
                p.requires_grad = False
            emb = self.model.get_input_embeddings()
            emb.weight.requires_grad = True
            allow_ids = list(self.step_token_ids) + [self.global_token_id]
            # Also allow the <STOP> token row to adapt, so one-pass global semantics
            # is not perturbed by a random frozen embedding when STOP appears in the prompt.
            try:
                allow_ids.append(int(self.stop_token_id))
            except Exception:
                pass
            step_ids = torch.as_tensor(allow_ids, dtype=torch.long, device=emb.weight.device)
            row_mask = torch.zeros((emb.weight.size(0), 1), dtype=emb.weight.dtype, device=emb.weight.device)
            row_mask[step_ids, 0] = 1.0

            def _mask_grad(grad):
                try:
                    return grad * row_mask
                except Exception:
                    return grad

            emb.weight.register_hook(_mask_grad)

            # Re-enable LoRA params (they may have been frozen by the global loop above).
            if _env_flag("VIGOR_LLM_LORA", "0") and self._lora_wrapped > 0:
                for n, p in self.model.named_parameters():
                    if isinstance(n, str) and ".lora_" in n:
                        p.requires_grad = True

    def train(self, mode: bool = True):
        # Keep the base LLM in eval() to avoid unintended dropout / training-time behavior.
        # LoRA params (and step token rows) still receive gradients in eval mode.
        super().train(mode)
        try:
            self.model.eval()
        except Exception:
            pass
        return self

    def encode_one_pass(
        self,
        utterances: List[str],
        step_texts: List[List[str]],
        *,
        sep: str = "\n",
        order_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Encode *global* + *step* representations in ONE causal-LM forward pass per sample.

        Inputs are teacher-forced (we already know step_texts):
          "{utterance} <sep> {phrase_1} <step1> <sep> ... {phrase_K} <stepK> [<cls>]"

        - `order_embeds[k]` is taken from the hidden state at the <stepk> token position.
        - If `memory_tokens>0`, append soft memory embeddings (no vocab expansion) and return:
            lang_embeds: [B, n_mem, out_dim]
            global_embed: lang_embeds[:,0,:]
          Else, append "<cls>" and return:
            global_embed: hidden at "<cls>" position.

        Returns:
          order_embeds: [B, order_len, out_dim]
          lang_embeds:  [B, n_mem, out_dim] or None
          global_embed: [B, out_dim]
        """
        B = len(utterances)
        if B == 0:
            device = next(self.parameters()).device
            return (
                torch.zeros((0, self.order_len, self.out_dim), device=device),
                None,
                torch.zeros((0, self.out_dim), device=device),
            )
        if len(step_texts) != B:
            raise ValueError(f"step_texts batch mismatch: len(step_texts)={len(step_texts)} B={B}")
        for i in range(B):
            if len(step_texts[i]) != self.order_len:
                raise ValueError(f"step_texts[{i}] length mismatch: expected {self.order_len} got={len(step_texts[i])}")

        max_total = int(self.cfg.max_length)
        text_max = max(1, max_total - int(self.n_mem))

        texts: List[str] = []
        # One-pass input mode:
        # - "teacher" (default): teacher-forced, includes per-step phrases.
        # - "pred": implicit chain generation, prompt contains ONLY utterance + <stepK> placeholders.
        onepass_mode = _env_str("VIGOR_LLM_ONEPASS_INPUT_MODE", "teacher").strip().lower()
        pred_mode = onepass_mode in {"pred", "predict", "planner", "plan", "latent"}

        # Truncation is designed for teacher-forced prompts (skip padded steps to keep context clean).
        # In pred-mode we keep all <stepK> placeholders so the model can learn STOP behavior.
        varlen_trunc = (not pred_mode) and _env_flag("VIGOR_VARLEN_ONEPASS_TRUNC", "0") and (order_valid_mask is not None)
        ovm = None
        if varlen_trunc and torch.is_tensor(order_valid_mask):
            try:
                ovm = order_valid_mask.detach().to(device="cpu")
            except Exception:
                ovm = None
        for i in range(B):
            u = str(utterances[i] or "").strip()
            parts: List[str] = [u] if u else []
            if pred_mode:
                # Implicit chain generation: no step phrases, only placeholders.
                # The hidden states at <stepK> are trained (via distillation/grounding) to represent each step.
                for k in range(self.order_len):
                    parts.append(f"<step{k+1}>")
            else:
                for k in range(self.order_len):
                    if varlen_trunc and ovm is not None:
                        try:
                            if float(ovm[i, k].item()) < 0.5:
                                continue
                        except Exception:
                            pass
                    phrase = str(step_texts[i][k] or "").strip()
                    tok = f"<step{k+1}>"
                    # Causal LM: put marker at the end of this step chunk.
                    parts.append(f"{phrase} {tok}".strip())
            if self.n_mem <= 0:
                # If no soft memory tokens, use a dedicated "<cls>" token for global semantics.
                parts.append(self.global_token)
            # Insert an explicit separator to reduce "step embedding collapse".
            texts.append((f"{sep} ".join([p for p in parts if p])).strip())

        def _tokenize_onepass(truncation_side: Optional[str] = None):
            old_side = getattr(self.tokenizer, "truncation_side", None)
            if truncation_side is not None and old_side is not None:
                try:
                    self.tokenizer.truncation_side = truncation_side
                except Exception:
                    old_side = None
            try:
                return self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=text_max,
                )
            finally:
                if truncation_side is not None and old_side is not None:
                    try:
                        self.tokenizer.truncation_side = old_side
                    except Exception:
                        pass

        enc = _tokenize_onepass(truncation_side=None)
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}
        input_ids = enc.get("input_ids", None)
        attn = enc.get("attention_mask", None)
        if input_ids is None or attn is None:
            raise RuntimeError("LLM tokenizer must return input_ids and attention_mask")

        # Robustness: if the tokenizer truncates the *end* of the prompt, later <stepK> markers
        # may be dropped (e.g., <step3>/<step4>), which breaks one-pass extraction.
        # Retry once using left-truncation to preserve tail markers.
        if _env_flag("VIGOR_LLM_ONEPASS_TRUNC_LEFT_FALLBACK", "1"):
            ovm_chk = None
            if varlen_trunc and ovm is not None:
                ovm_chk = ovm

            def _missing_any_step() -> bool:
                for i in range(B):
                    for k, step_id in enumerate(self.step_token_ids):
                        if ovm_chk is not None:
                            try:
                                if float(ovm_chk[i, k].item()) < 0.5:
                                    continue
                            except Exception:
                                pass
                        if (input_ids[i] == int(step_id)).any():
                            continue
                        return True
                return False

            if _missing_any_step():
                enc2 = _tokenize_onepass(truncation_side="left")
                enc2 = {k: v.to(device=device) for k, v in enc2.items()}
                input_ids2 = enc2.get("input_ids", None)
                attn2 = enc2.get("attention_mask", None)
                if input_ids2 is None or attn2 is None:
                    raise RuntimeError("LLM tokenizer must return input_ids and attention_mask")
                input_ids, attn, enc = input_ids2, attn2, enc2

        base_emb = self.model.get_input_embeddings()(input_ids)
        base_len = int(base_emb.size(1))

        inputs_embeds = base_emb
        if self.n_mem > 0:
            if self.mem_tokens is None:
                raise RuntimeError("cfg.memory_tokens>0 but mem_tokens is None")
            mem = self.mem_tokens.to(device=base_emb.device, dtype=base_emb.dtype).unsqueeze(0).expand(B, self.n_mem, -1)
            inputs_embeds = torch.cat([base_emb, mem], dim=1)
            attn_mem = torch.ones((B, self.n_mem), device=attn.device, dtype=attn.dtype)
            attn = torch.cat([attn, attn_mem], dim=1)

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]

        # Gather step hidden states from the *base* token positions.
        step_h = torch.zeros((B, self.order_len, hs.size(-1)), device=hs.device, dtype=hs.dtype)
        ovm_dev = None
        if varlen_trunc and torch.is_tensor(order_valid_mask):
            ovm_dev = order_valid_mask.to(device=hs.device)
        for i in range(B):
            for k, step_id in enumerate(self.step_token_ids):
                where = (input_ids[i] == int(step_id)).nonzero(as_tuple=False).view(-1)
                if where.numel() == 0:
                    if varlen_trunc and (ovm_dev is not None):
                        try:
                            if float(ovm_dev[i, k].item()) < 0.5:
                                continue
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Missing <step{k+1}> token in one-pass LLM input (sample={i}). "
                        f"Increase VIGOR_LLM_MAX_LEN or shorten the input text "
                        f"(text_max={text_max}, n_mem={int(self.n_mem)})."
                    )
                pos = int(where[-1].item())
                step_h[i, k] = hs[i, pos]

        step_h = step_h.to(dtype=self.proj_step[0].weight.dtype)
        order_embeds = self.proj_step(step_h)

        lang_embeds: Optional[torch.Tensor] = None
        if self.n_mem > 0:
            if self.proj_mem is None:
                raise RuntimeError("cfg.memory_tokens>0 but proj_mem is None")
            mem_h = hs[:, base_len : base_len + self.n_mem, :].to(dtype=self.proj_mem[0].weight.dtype)
            lang_embeds = self.proj_mem(mem_h)
            global_embed = lang_embeds[:, 0, :]
        else:
            # Global embed from "<cls>" token (must exist in base input_ids).
            cls_h = torch.zeros((B, hs.size(-1)), device=hs.device, dtype=hs.dtype)
            for i in range(B):
                where = (input_ids[i] == int(self.global_token_id)).nonzero(as_tuple=False).view(-1)
                if where.numel() == 0:
                    raise RuntimeError(
                        f"Missing global <cls> token in one-pass LLM input (sample={i}). "
                        f"Increase VIGOR_LLM_MAX_LEN or shorten the input text."
                    )
                pos = int(where[-1].item())
                cls_h[i] = hs[i, pos]
            cls_h = cls_h.to(dtype=self.proj_global[0].weight.dtype)
            global_embed = self.proj_global(cls_h)

        return order_embeds, lang_embeds, global_embed

    def _hidden_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        for k in ("hidden_size", "n_embd", "d_model"):
            if cfg is not None and getattr(cfg, k, None) is not None:
                return int(getattr(cfg, k))
        raise RuntimeError("Cannot infer LLM hidden size from config")

    def forward(self, step_texts: List[List[str]]) -> torch.Tensor:
        """
        Args:
          step_texts: list length B, each is list length order_len
        Returns:
          order_embeds: [B, order_len, out_dim]
        """
        B = len(step_texts)
        if B == 0:
            return torch.zeros((0, self.order_len, self.out_dim), device=next(self.parameters()).device)
        for i in range(B):
            if len(step_texts[i]) != self.order_len:
                raise ValueError(
                    f"step_texts[{i}] length mismatch: expected {self.order_len} got={len(step_texts[i])}"
                )

        texts: List[str] = []
        for i in range(B):
            for k in range(self.order_len):
                s = str(step_texts[i][k] or "").strip()
                tok = f"<step{k+1}>"
                # IMPORTANT (causal LLM):
                # place the marker at the end so its hidden state can attend to the step phrase.
                texts.append(f"{s} {tok}".strip())

        max_len = int(self.cfg.max_length)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}

        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        attn = enc.get("attention_mask", None)
        if attn is None:
            raise RuntimeError("LLM tokenizer did not return attention_mask")
        last_pos = attn.long().sum(dim=-1) - 1
        idx = torch.arange(hs.size(0), device=hs.device)
        step_h = hs[idx, last_pos]
        step_h = step_h.to(dtype=self.proj_step[0].weight.dtype)
        proj = self.proj_step(step_h)
        proj = proj.reshape(B, self.order_len, self.out_dim)
        return proj

    def encode_multi_pass_pred(
        self,
        utterances: List[str],
        *,
        order_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-pass analogue of one-pass pred-mode (implicit chain generation).

        For each sample i and each step k, run an isolated forward:
          - valid step:   "{utterance} <step{k}>"
          - invalid step: "<STOP> <step{k}>"   (or "{utterance} <STOP> <step{k}>" if configured)

        We read the hidden state at the final "<step{k}>" marker as the k-th step embedding.
        """
        B = len(utterances)
        if B == 0:
            return torch.zeros((0, self.order_len, self.out_dim), device=next(self.parameters()).device)

        invalid_use_stop_only = _env_flag("VIGOR_VARLEN_MULTIPASS_INVALID_STOP_ONLY", "1")
        ovm = None
        if torch.is_tensor(order_valid_mask):
            try:
                ovm = order_valid_mask.detach().to(device="cpu")
            except Exception:
                ovm = None

        texts: List[str] = []
        for i in range(B):
            u = str(utterances[i] or "").strip()
            for k in range(self.order_len):
                valid = True
                if ovm is not None:
                    try:
                        valid = float(ovm[i, k].item()) >= 0.5
                    except Exception:
                        valid = True
                tok = f"<step{k+1}>"
                if not valid:
                    base = self.stop_token if invalid_use_stop_only else f"{u} {self.stop_token}".strip()
                else:
                    base = u
                texts.append(f"{base} {tok}".strip())

        max_len = int(self.cfg.max_length)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}

        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        attn = enc.get("attention_mask", None)
        if attn is None:
            raise RuntimeError("LLM tokenizer did not return attention_mask")
        last_pos = attn.long().sum(dim=-1) - 1
        idx = torch.arange(hs.size(0), device=hs.device)
        step_h = hs[idx, last_pos]
        step_h = step_h.to(dtype=self.proj_step[0].weight.dtype)
        proj = self.proj_step(step_h)
        proj = proj.reshape(B, self.order_len, self.out_dim)
        return proj

    def encode_multi_pass_teacher(
        self,
        utterances: List[str],
        step_texts: List[List[str]],
        *,
        order_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-pass teacher-forced step encoding.

        For each sample i and each step k, run an isolated forward:
          - valid step:   "{utterance} {phrase_k} <step{k}>"
          - invalid step: "<STOP> <step{k}>"   (or "{utterance} <STOP> <step{k}>" if configured)

        This is the multi-pass counterpart of one-pass "teacher" mode (which concatenates all steps).
        """
        B = len(utterances)
        if B == 0:
            return torch.zeros((0, self.order_len, self.out_dim), device=next(self.parameters()).device)
        if len(step_texts) != B:
            raise ValueError(f"step_texts batch mismatch: len(step_texts)={len(step_texts)} B={B}")
        for i in range(B):
            if len(step_texts[i]) != self.order_len:
                raise ValueError(f"step_texts[{i}] length mismatch: expected {self.order_len} got={len(step_texts[i])}")

        invalid_use_stop_only = _env_flag("VIGOR_VARLEN_MULTIPASS_INVALID_STOP_ONLY", "1")
        include_utt = _env_flag("VIGOR_LLM_MULTIPASS_TEACHER_INCLUDE_UTT", "1")

        ovm = None
        if torch.is_tensor(order_valid_mask):
            try:
                ovm = order_valid_mask.detach().to(device="cpu")
            except Exception:
                ovm = None

        texts: List[str] = []
        for i in range(B):
            u = str(utterances[i] or "").strip()
            for k in range(self.order_len):
                valid = True
                if ovm is not None:
                    try:
                        valid = float(ovm[i, k].item()) >= 0.5
                    except Exception:
                        valid = True
                tok = f"<step{k+1}>"
                if not valid:
                    base = self.stop_token if invalid_use_stop_only else f"{u} {self.stop_token}".strip()
                else:
                    phrase = str(step_texts[i][k] or "").strip()
                    if include_utt and u:
                        base = f"{u} {phrase}".strip()
                    else:
                        base = phrase
                texts.append(f"{base} {tok}".strip())

        max_len = int(self.cfg.max_length)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}

        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        attn = enc.get("attention_mask", None)
        if attn is None:
            raise RuntimeError("LLM tokenizer did not return attention_mask")
        last_pos = attn.long().sum(dim=-1) - 1
        idx = torch.arange(hs.size(0), device=hs.device)
        step_h = hs[idx, last_pos]
        step_h = step_h.to(dtype=self.proj_step[0].weight.dtype)
        proj = self.proj_step(step_h)
        proj = proj.reshape(B, self.order_len, self.out_dim)
        return proj

    def encode_global(self, utterances: List[str]) -> torch.Tensor:
        """
        Encode a *global* semantic vector for each utterance, aligned to BERT [CLS].
        We append a dedicated "<cls>" token at the end so its hidden state can attend to the whole utterance.
        Returns: [B, out_dim]
        """
        B = len(utterances)
        if B == 0:
            return torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
        texts = [f"{str(u or '').strip()} {self.global_token}".strip() for u in utterances]
        max_len = int(self.cfg.max_length)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        attn = enc.get("attention_mask", None)
        if attn is None:
            raise RuntimeError("LLM tokenizer did not return attention_mask")
        last_pos = attn.long().sum(dim=-1) - 1
        idx = torch.arange(hs.size(0), device=hs.device)
        h = hs[idx, last_pos]
        h = h.to(dtype=self.proj_global[0].weight.dtype)
        return self.proj_global(h)

    def encode_lang_embeds(self, utterances: List[str]) -> torch.Tensor:
        """
        Encode a *token-level* memory for each utterance using soft memory tokens.

        We do NOT expand the tokenizer vocabulary. Instead, we append `n_mem` learnable embeddings
        to the end of the utterance `inputs_embeds`, and read their final hidden states as memory.

        Returns:
          lang_embeds: [B, n_mem, out_dim]
        """
        if self.n_mem <= 0 or self.mem_tokens is None or self.proj_mem is None:
            raise RuntimeError("encode_lang_embeds requires cfg.memory_tokens > 0")
        B = len(utterances)
        if B == 0:
            return torch.zeros((0, self.n_mem, self.out_dim), device=next(self.parameters()).device)

        max_total = int(self.cfg.max_length)
        text_max = max(1, max_total - int(self.n_mem))
        texts = [str(u or "").strip() for u in utterances]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max,
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device=device) for k, v in enc.items()}
        input_ids = enc.get("input_ids", None)
        attn = enc.get("attention_mask", None)
        if input_ids is None or attn is None:
            raise RuntimeError("LLM tokenizer must return input_ids and attention_mask")

        base_emb = self.model.get_input_embeddings()(input_ids)
        # Append learnable memory embeddings at the end.
        mem = self.mem_tokens.to(device=base_emb.device, dtype=base_emb.dtype).unsqueeze(0).expand(B, self.n_mem, -1)
        inputs_embeds = torch.cat([base_emb, mem], dim=1)
        attn_mem = torch.ones((B, self.n_mem), device=attn.device, dtype=attn.dtype)
        attn = torch.cat([attn, attn_mem], dim=1)

        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1]
        mem_h = hs[:, -self.n_mem :, :].to(dtype=self.proj_mem[0].weight.dtype)
        return self.proj_mem(mem_h)


class ReferIt3DNetTransformerLlamaStepSlot(nn.Module):
    """
    Wrapper around Vigor's `ReferIt3DNet_transformer`:
    - Listener is typically initialized from a BERT-step-slot checkpoint and frozen.
    - LLM produces per-step `order_embeds` in the same space as BERT step slots.
    - Optionally adds a distillation loss between LLM order_embeds and BERT step-slot embeddings.
    """

    def __init__(self, listener: nn.Module, llm: LlamaStepSlotOrderEncoder, cfg: LlamaStepSlotConfig):
        super().__init__()
        self.listener = listener
        self.llm = llm
        self.cfg = cfg
        self.order_len = int(cfg.order_len)
        self.last_distill_step: Optional[float] = None
        self.last_distill_global: Optional[float] = None
        self.last_distill_step_mp: Optional[float] = None
        self.last_distill_global_mp: Optional[float] = None
        self.last_mp_op_distill: Optional[float] = None
        self._train_listener_parts = [
            p.strip().lower()
            for p in _env_str("VIGOR_LLM_STEPSLOT_TRAIN_LISTENER_PARTS", "").split(",")
            if p.strip()
        ]
        # A learnable STOP representation in the same space as `order_embeds` (Vigor inner_dim).
        # When `VIGOR_VARLEN_CHAIN=1`, padded slots can be explicitly mapped to this vector.
        stop_dim = int(getattr(self.llm, "out_dim", 0) or 0)
        if stop_dim <= 0:
            stop_dim = int(getattr(self.listener, "inner_dim", 768) or 768)
        self.stop_embed = nn.Parameter(torch.zeros(stop_dim))
        nn.init.normal_(self.stop_embed, mean=0.0, std=0.02)

        # A lightweight variable-length (STOP) gate head.
        #
        # It predicts a per-step STOP probability from `order_embeds` and can be used to:
        #  - compute an auxiliary gate loss during training
        #  - decide chain length online at inference without relying on a hand-picked cosine threshold
        gate_hidden = max(64, int(stop_dim) // 4)
        self.varlen_gate = nn.Sequential(
            nn.LayerNorm(stop_dim),
            nn.Linear(stop_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.last_gate_loss: Optional[float] = None

        # Default: keep listener frozen (we only learn LLM step rows + projection).
        # Optional: allow training ONLY some lightweight listener heads in Phase-B,
        # or train ONLY the listener for ablations (see VIGOR_TRAIN_LISTENER_ONLY).
        train_listener_only = _env_flag("VIGOR_TRAIN_LISTENER_ONLY", "0") or _env_flag(
            "VIGOR_LLM_STEPSLOT_TRAIN_LISTENER_ONLY", "0"
        )
        train_listener = train_listener_only or _env_flag("VIGOR_LLM_STEPSLOT_TRAIN_LISTENER", "0")
        if not train_listener:
            for p in self.listener.parameters():
                p.requires_grad = False
            self.listener.eval()
        else:
            # If parts are specified, freeze everything first, then selectively unfreeze.
            if self._train_listener_parts:
                for p in self.listener.parameters():
                    p.requires_grad = False
                parts = set(self._train_listener_parts)
                # Supported parts (comma-separated):
                #   - language_clf / lang_clf
                #   - anchor_clf / anchor
                if ("language_clf" in parts) or ("lang_clf" in parts) or ("lang" in parts):
                    if hasattr(self.listener, "language_clf"):
                        for p in self.listener.language_clf.parameters():
                            p.requires_grad = True
                if ("anchor_clf" in parts) or ("anchor" in parts):
                    if hasattr(self.listener, "anchor_clf"):
                        for p in self.listener.anchor_clf.parameters():
                            p.requires_grad = True
                # Keep the rest frozen and stable.
                self.listener.eval()

        # Optional: train ONLY the varlen gate head (protects grounding by avoiding any LLM-side drift).
        train_only_gate = _env_flag("VIGOR_VARLEN_GATE_TRAIN_ONLY", "0")
        if train_only_gate:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.varlen_gate.parameters():
                p.requires_grad = True
            # Optionally also train stop_embed (OFF by default).
            if _env_flag("VIGOR_VARLEN_GATE_TRAIN_STOP_EMBED", "0"):
                self.stop_embed.requires_grad = True

        # Optional: train ONLY the listener (for "naive end-to-end finetune" ablations).
        # This freezes the entire LLM-side interface (step rows, LoRA, projections, STOP/gate),
        # and unfreezes the Vigor listener end-to-end.
        #
        # NOTE: we intentionally keep this as an env-gated behavior to avoid changing defaults.
        if train_listener_only and (not train_only_gate):
            for p in self.parameters():
                p.requires_grad = False
            for p in self.listener.parameters():
                p.requires_grad = True

        # BERT tokenizer for teacher distillation (step + global).
        try:
            from transformers import BertTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError("transformers is required for ReferIt3DNetTransformerLlamaStepSlot") from e
        bert_path = getattr(self.listener, "bert_pretrain_path", None)
        if not bert_path:
            raise RuntimeError("listener.bert_pretrain_path is missing")
        self.bert_tokenizer = BertTokenizer.from_pretrained(str(bert_path))
        step_tokens = [f"<step{i+1}>" for i in range(self.order_len)]
        _ = self.bert_tokenizer.add_special_tokens({"additional_special_tokens": step_tokens})

    def train(self, mode: bool = True):
        super().train(mode)
        train_listener = _env_flag("VIGOR_TRAIN_LISTENER_ONLY", "0") or _env_flag(
            "VIGOR_LLM_STEPSLOT_TRAIN_LISTENER_ONLY", "0"
        ) or _env_flag("VIGOR_LLM_STEPSLOT_TRAIN_LISTENER", "0")
        if not train_listener:
            # Keep frozen listener in eval mode.
            self.listener.eval()
            # If we only train the varlen gate, keep the LLM side in eval too (reduces dropout noise).
            if _env_flag("VIGOR_VARLEN_GATE_TRAIN_ONLY", "0"):
                try:
                    self.llm.eval()
                except Exception:
                    pass
        else:
            # If only training selected heads, keep the full listener in eval to avoid
            # injecting dropout noise into the (frozen) refer encoder; only heads go train.
            if self._train_listener_parts:
                self.listener.eval()
                parts = set(self._train_listener_parts)
                if hasattr(self.listener, "language_clf") and (("language_clf" in parts) or ("lang_clf" in parts) or ("lang" in parts)):
                    self.listener.language_clf.train(mode)
                if hasattr(self.listener, "anchor_clf") and (("anchor_clf" in parts) or ("anchor" in parts)):
                    self.listener.anchor_clf.train(mode)
        return self

    @torch.no_grad()
    def _teacher_step_embeds(self, batch: dict) -> torch.Tensor:
        """
        Teacher = BERT step-marker hidden at token position 1 (after [CLS]).
        Shape: [B, order_len, D]
        """
        ref_order = batch.get("referential_order", None)
        B = int(batch["target_pos"].size(0))
        texts: List[str] = []
        for i in range(B):
            for k in range(self.order_len):
                phrase = str(_safe_get_referential_token(ref_order, i, k) or "").strip()
                texts.append(f"<step{k+1}> {phrase}".strip())

        device = next(self.listener.parameters()).device
        enc = self.bert_tokenizer(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(device=device) for k, v in enc.items()}
        hs = self.listener.language_encoder(**enc)[0]  # [B*O,L,D]
        hs = hs.reshape(B, self.order_len, hs.size(1), hs.size(2))

        pos = 1 if hs.size(2) > 1 else 0  # <stepK> should be at pos=1
        t = hs[:, :, pos, :].contiguous()
        if int(t.size(1)) != int(self.order_len):
            raise RuntimeError(f"teacher order_len mismatch: got {int(t.size(1))} expected {int(self.order_len)}")
        return t

    @torch.no_grad()
    def _teacher_global_embed(self, batch: dict) -> torch.Tensor:
        """
        Teacher global semantics = BERT [CLS] on the utterance (lang_tokens).
        Shape: [B, D]
        """
        lang_tokens = batch.get("lang_tokens", None)
        if not isinstance(lang_tokens, dict) or "input_ids" not in lang_tokens:
            raise RuntimeError("batch.lang_tokens is required for global distillation")
        device = next(self.listener.parameters()).device
        lang_tokens = {k: v.to(device=device) for k, v in lang_tokens.items()}
        hs = self.listener.language_encoder(**lang_tokens)[0]
        return hs[:, 0, :].contiguous()

    @torch.no_grad()
    def mk_only_forward(self, batch: dict) -> None:
        """
        Lightweight forward for varlen m_k probing.

        It computes and caches:
          - self.last_order_embeds (pre STOP replacement)
          - self.last_order_embeds_post_stop (post STOP replacement if enabled)
          - self.last_order_valid_mask (if `ori_order_len` is available)

        It intentionally skips the ViGOR listener / grounding head.
        """
        tokens = batch.get("tokens", None)
        if not isinstance(tokens, list) or len(tokens) == 0:
            raise RuntimeError("mk_only_forward requires batch.tokens as a non-empty list[str]")
        B = len(tokens)

        ref_order = batch.get("referential_order", None)
        step_texts: List[List[str]] = []
        for i in range(B):
            row = []
            for k in range(self.order_len):
                row.append(str(_safe_get_referential_token(ref_order, i, k) or "").strip())
            step_texts.append(row)

        # Build order_valid_mask if the dataset provides `ori_order_len` (even if varlen gating is off).
        order_valid_mask = _maybe_build_order_valid_mask(batch, self.order_len, device=next(self.parameters()).device)

        use_onepass = _env_flag("VIGOR_LLM_STEPSLOT_ONEPASS", "0")

        def _encode_multipass(tokens_: List[str]) -> torch.Tensor:
            multipass_mode = _env_str("VIGOR_LLM_MULTIPASS_INPUT_MODE", "teacher").strip().lower()
            pred_multipass = multipass_mode in {"pred", "predict", "planner", "plan", "latent"}
            if pred_multipass:
                return self.llm.encode_multi_pass_pred(tokens_, order_valid_mask=order_valid_mask)
            if _env_flag("VIGOR_LLM_MULTIPASS_TEACHER_LEGACY", "0"):
                return self.llm(step_texts)
            return self.llm.encode_multi_pass_teacher(tokens_, step_texts, order_valid_mask=order_valid_mask)

        if use_onepass:
            order_embeds, _, _ = self.llm.encode_one_pass(tokens, step_texts, order_valid_mask=order_valid_mask)
        else:
            order_embeds = _encode_multipass(tokens)

        # Cache pre-STOP-replacement embeddings + mask.
        try:
            self.last_order_embeds = order_embeds.detach()
        except Exception:
            self.last_order_embeds = None
        try:
            self.last_order_valid_mask = order_valid_mask.detach() if torch.is_tensor(order_valid_mask) else None
        except Exception:
            self.last_order_valid_mask = None

        # Optional replacement (kept for parity with full forward; should be OFF for honest m_k probing).
        stop_replace = _env_flag("VIGOR_STOP_EMBED_REPLACE", "0")
        varlen_enabled = _env_flag("VIGOR_VARLEN_CHAIN", "0")
        if varlen_enabled and (order_valid_mask is not None) and stop_replace:
            try:
                m = order_valid_mask.to(device=order_embeds.device, dtype=order_embeds.dtype).unsqueeze(-1)  # [B,O,1]
                tgt = self.stop_embed.to(device=order_embeds.device, dtype=order_embeds.dtype).view(1, 1, -1)
                order_embeds = order_embeds * m + tgt * (1.0 - m)
            except Exception:
                pass

        try:
            self.last_order_embeds_post_stop = order_embeds.detach()
        except Exception:
            self.last_order_embeds_post_stop = None

    def _varlen_gate_logits(self, order_embeds: torch.Tensor) -> torch.Tensor:
        """
        Return per-step STOP logits from `order_embeds`.
        Shape: [B,O]
        """
        if (not torch.is_tensor(order_embeds)) or order_embeds.dim() != 3:
            raise RuntimeError("order_embeds must be [B,O,D]")
        x = order_embeds
        if _env_flag("VIGOR_VARLEN_GATE_DETACH_EMB", "0"):
            x = x.detach()
        logits = self.varlen_gate(x).squeeze(-1)
        return logits

    def forward(self, batch: dict, epoch=None):
        if "target_pos" not in batch:
            raise RuntimeError("batch must contain target_pos")
        B = int(batch["target_pos"].size(0))
        tokens = batch.get("tokens", None)
        if not isinstance(tokens, list) or len(tokens) != B:
            raise RuntimeError(
                f"Expected batch.tokens to be a list of length B={B} (DataParallel unsupported); got {type(tokens)} "
                f"len={len(tokens) if isinstance(tokens, list) else 'n/a'}"
            )

        ref_order = batch.get("referential_order", None)
        step_texts: List[List[str]] = []
        for i in range(B):
            row = []
            for k in range(self.order_len):
                row.append(str(_safe_get_referential_token(ref_order, i, k) or "").strip())
            step_texts.append(row)

        varlen_enabled = _env_flag("VIGOR_VARLEN_CHAIN", "0")
        varlen_mask_source = _env_str("VIGOR_VARLEN_MASK_SOURCE", "oracle").strip().lower()
        varlen_pred_tau = _env_float("VIGOR_VARLEN_PRED_TAU", 0.90)
        varlen_cp_min_score = _env_float("VIGOR_VARLEN_CP_MIN_SCORE", 0.05)
        order_valid_mask = None
        if varlen_enabled:
            order_valid_mask = _maybe_build_order_valid_mask(batch, self.order_len, device=next(self.parameters()).device)
        # Keep an oracle copy for auxiliary supervision (even if we later override with predicted masks).
        order_valid_mask_oracle = order_valid_mask

        use_onepass = _env_flag("VIGOR_LLM_STEPSLOT_ONEPASS", "0")
        joint = _env_flag("VIGOR_LLM_JOINT_MP_OP", "0")
        listener_src = _env_str("VIGOR_LLM_LISTENER_ORDER_SOURCE", "auto").strip().lower()

        def _encode_multipass(tokens_: List[str]) -> torch.Tensor:
            multipass_mode = _env_str("VIGOR_LLM_MULTIPASS_INPUT_MODE", "teacher").strip().lower()
            pred_multipass = multipass_mode in {"pred", "predict", "planner", "plan", "latent"}
            if pred_multipass:
                return self.llm.encode_multi_pass_pred(tokens_, order_valid_mask=order_valid_mask)
            if _env_flag("VIGOR_LLM_MULTIPASS_TEACHER_LEGACY", "0"):
                return self.llm(step_texts)
            return self.llm.encode_multi_pass_teacher(tokens_, step_texts, order_valid_mask=order_valid_mask)

        order_embeds_op = None
        order_embeds_mp = None
        lang_embeds = None
        global_embed = None

        if use_onepass:
            order_embeds_op, lang_embeds, global_embed = self.llm.encode_one_pass(
                tokens, step_texts, order_valid_mask=order_valid_mask
            )
            if joint:
                order_embeds_mp = _encode_multipass(tokens)
        else:
            order_embeds_mp = _encode_multipass(tokens)
            if joint:
                order_embeds_op, lang_embeds, global_embed = self.llm.encode_one_pass(
                    tokens, step_texts, order_valid_mask=order_valid_mask
                )

        # Global/lang embeddings (for listener) come from the one-pass path if present; otherwise from a standalone pass.
        if global_embed is None:
            if getattr(self.llm, "n_mem", 0) > 0:
                lang_embeds = self.llm.encode_lang_embeds(tokens)
                global_embed = lang_embeds[:, 0, :]
            else:
                global_embed = self.llm.encode_global(tokens)

        # Which order embeddings should drive grounding?
        # - auto: follow use_onepass unless joint wants to force OP as primary.
        # - op / onepass: always use OP (requires it exists)
        # - mp / multipass: always use MP (requires it exists)
        if listener_src in {"op", "onepass"}:
            if order_embeds_op is None:
                raise RuntimeError("VIGOR_LLM_LISTENER_ORDER_SOURCE=op but one-pass order_embeds not available")
            order_embeds = order_embeds_op
        elif listener_src in {"mp", "multipass"}:
            if order_embeds_mp is None:
                raise RuntimeError("VIGOR_LLM_LISTENER_ORDER_SOURCE=mp but multi-pass order_embeds not available")
            order_embeds = order_embeds_mp
        else:
            # auto
            order_embeds = order_embeds_op if order_embeds_op is not None else order_embeds_mp
            if order_embeds is None:
                raise RuntimeError("Failed to compute order_embeds (neither one-pass nor multi-pass available)")

        # Expose the embeddings and masks for downstream probing (e.g. STOP-based m_k prediction).
        # Note: these are the *pre-STOP-replacement* embeddings.
        try:
            self.last_order_embeds = order_embeds.detach()
        except Exception:
            self.last_order_embeds = None
        try:
            self.last_order_valid_mask = order_valid_mask.detach() if torch.is_tensor(order_valid_mask) else None
        except Exception:
            self.last_order_valid_mask = None

        # Optionally override the oracle mask with a predicted mask for true online varlen gating.
        # - pred/cp: cosine similarity to stop_embed
        # - gate: learned STOP gate head
        if varlen_enabled and varlen_mask_source in {"pred", "predict", "stop", "changepoint", "cp", "gate"}:
            try:
                if varlen_mask_source == "gate":
                    tau_gate = _env_float("VIGOR_VARLEN_GATE_TAU", 0.5)
                    gate_decode = _env_str("VIGOR_VARLEN_GATE_DECODE", "threshold").strip().lower()
                    gate_mono = _env_flag("VIGOR_VARLEN_GATE_MONO", "0")
                    prob = torch.sigmoid(self._varlen_gate_logits(order_embeds))
                    order_valid_mask = _predict_order_valid_mask_from_gate_probs(
                        prob, tau=float(tau_gate), mode=gate_decode, monotonic=bool(gate_mono)
                    )
                elif varlen_mask_source in {"changepoint", "cp"}:
                    order_valid_mask = _predict_order_valid_mask_changepoint_from_stop(
                        order_embeds, self.stop_embed, min_score=float(varlen_cp_min_score)
                    )
                else:
                    order_valid_mask = _predict_order_valid_mask_from_stop(
                        order_embeds,
                        self.stop_embed,
                        tau=float(varlen_pred_tau),
                    )
            except Exception:
                # Keep eval robust: if prediction fails, fall back to oracle/no mask.
                pass
            try:
                self.last_order_valid_mask_pred = order_valid_mask.detach() if torch.is_tensor(order_valid_mask) else None
            except Exception:
                self.last_order_valid_mask_pred = None

        distill_w = _env_float("VIGOR_LLM_DISTILL_W", float(self.cfg.distill_w))
        distill = None
        teacher = None
        if (distill_w > 0) or (_env_float("VIGOR_LLM_DISTILL_W_MP", 0.0) > 0) or (_env_float("VIGOR_LLM_MP_OP_DISTILL_W", 0.0) > 0):
            teacher = self._teacher_step_embeds(batch)

        if distill_w > 0 and teacher is not None:
            if varlen_enabled and (order_valid_mask is not None):
                m = order_valid_mask.to(device=order_embeds.device, dtype=torch.float32)  # [B,O]
                denom = m.sum().clamp(min=1.0)
                if self.cfg.distill_type == "mse":
                    per = (order_embeds.float() - teacher.float()).pow(2).mean(dim=-1)  # [B,O]
                else:
                    a = F.normalize(order_embeds.float(), dim=-1)
                    b = F.normalize(teacher.float(), dim=-1)
                    per = (1.0 - (a * b).sum(dim=-1))  # [B,O]
                distill = (per * m).sum() / denom
            else:
                if self.cfg.distill_type == "mse":
                    distill = F.mse_loss(order_embeds.float(), teacher.float())
                else:
                    a = F.normalize(order_embeds.float(), dim=-1)
                    b = F.normalize(teacher.float(), dim=-1)
                    distill = (1.0 - (a * b).sum(dim=-1)).mean()

        g_w = _env_float("VIGOR_LLM_GLOBAL_DISTILL_W", float(self.cfg.global_distill_w))
        g_type = _env_str("VIGOR_LLM_GLOBAL_DISTILL_TYPE", str(self.cfg.global_distill_type)).lower()
        distill_g = None
        if g_w > 0:
            teacher_g = self._teacher_global_embed(batch)
            if g_type == "mse":
                distill_g = F.mse_loss(global_embed.float(), teacher_g.float())
            else:
                a = F.normalize(global_embed.float(), dim=-1)
                b = F.normalize(teacher_g.float(), dim=-1)
                distill_g = (1.0 - (a * b).sum(dim=-1)).mean()

        # Optional: extra distillation for the *multi-pass* branch (useful when listener is driven by OP).
        distill_mp = None
        distill_g_mp = None
        distill_w_mp = _env_float("VIGOR_LLM_DISTILL_W_MP", 0.0)
        g_w_mp = _env_float("VIGOR_LLM_GLOBAL_DISTILL_W_MP", 0.0)
        if (order_embeds_mp is not None) and (teacher is not None) and (distill_w_mp > 0):
            if varlen_enabled and (order_valid_mask is not None):
                m = order_valid_mask.to(device=order_embeds_mp.device, dtype=torch.float32)
                denom = m.sum().clamp(min=1.0)
                if self.cfg.distill_type == "mse":
                    per = (order_embeds_mp.float() - teacher.float()).pow(2).mean(dim=-1)
                else:
                    a = F.normalize(order_embeds_mp.float(), dim=-1)
                    b = F.normalize(teacher.float(), dim=-1)
                    per = (1.0 - (a * b).sum(dim=-1))
                distill_mp = (per * m).sum() / denom
            else:
                if self.cfg.distill_type == "mse":
                    distill_mp = F.mse_loss(order_embeds_mp.float(), teacher.float())
                else:
                    a = F.normalize(order_embeds_mp.float(), dim=-1)
                    b = F.normalize(teacher.float(), dim=-1)
                    distill_mp = (1.0 - (a * b).sum(dim=-1)).mean()

        if g_w_mp > 0:
            try:
                teacher_g = self._teacher_global_embed(batch)
                if g_type == "mse":
                    distill_g_mp = F.mse_loss(global_embed.float(), teacher_g.float())
                else:
                    a = F.normalize(global_embed.float(), dim=-1)
                    b = F.normalize(teacher_g.float(), dim=-1)
                    distill_g_mp = (1.0 - (a * b).sum(dim=-1)).mean()
            except Exception:
                distill_g_mp = None

        # Joint MP->OP distillation: pull the one-pass (student) towards the isolated multi-pass (teacher).
        mp_op = None
        mp_op_w = _env_float("VIGOR_LLM_MP_OP_DISTILL_W", 0.0)
        mp_op_type = _env_str("VIGOR_LLM_MP_OP_DISTILL_TYPE", "cos").strip().lower()
        if (mp_op_w > 0) and (order_embeds_op is not None) and (order_embeds_mp is not None):
            try:
                if varlen_enabled and (order_valid_mask is not None):
                    m = order_valid_mask.to(device=order_embeds_op.device, dtype=torch.float32)
                    denom = m.sum().clamp(min=1.0)
                    if mp_op_type == "mse":
                        per = (order_embeds_op.float() - order_embeds_mp.float()).pow(2).mean(dim=-1)
                    else:
                        a = F.normalize(order_embeds_op.float(), dim=-1)
                        b = F.normalize(order_embeds_mp.float(), dim=-1)
                        per = (1.0 - (a * b).sum(dim=-1))
                    mp_op = (per * m).sum() / denom
                else:
                    if mp_op_type == "mse":
                        mp_op = F.mse_loss(order_embeds_op.float(), order_embeds_mp.float())
                    else:
                        a = F.normalize(order_embeds_op.float(), dim=-1)
                        b = F.normalize(order_embeds_mp.float(), dim=-1)
                        mp_op = (1.0 - (a * b).sum(dim=-1)).mean()
            except Exception:
                mp_op = None

        # Expose latest distillation losses for outer training loops / tqdm postfix.
        try:
            self.last_distill_step = float(distill.item()) if distill is not None else None
        except Exception:
            self.last_distill_step = None
        try:
            self.last_distill_global = float(distill_g.item()) if distill_g is not None else None
        except Exception:
            self.last_distill_global = None
        try:
            self.last_distill_step_mp = float(distill_mp.item()) if distill_mp is not None else None
        except Exception:
            self.last_distill_step_mp = None
        try:
            self.last_distill_global_mp = float(distill_g_mp.item()) if distill_g_mp is not None else None
        except Exception:
            self.last_distill_global_mp = None
        try:
            self.last_mp_op_distill = float(mp_op.item()) if mp_op is not None else None
        except Exception:
            self.last_mp_op_distill = None

        # Inject LLM order embeddings into listener forward.
        # Important: avoid silent fallback to BERT order_tokens.
        batch = dict(batch)
        stop_w = _env_float("VIGOR_STOP_EMBED_W", 0.0)
        stop_replace = _env_flag("VIGOR_STOP_EMBED_REPLACE", "0")

        # Optional STOP regularization: make the "inactive" slots converge to a consistent STOP vector.
        stop_reg = None
        if varlen_enabled and (order_valid_mask is not None) and stop_w > 0:
            try:
                stop_mask = (order_valid_mask.to(device=order_embeds.device) < 0.5)  # [B,O] bool
                denom = stop_mask.float().sum().clamp(min=1.0)
                tgt = self.stop_embed.to(device=order_embeds.device, dtype=order_embeds.dtype).view(1, 1, -1)
                diff = (order_embeds - tgt).pow(2).mean(dim=-1)  # [B,O]
                stop_reg = (diff * stop_mask.float()).sum() / denom
            except Exception:
                stop_reg = None

        # Optional variable-length gate loss: train a small head to predict STOP/valid per step.
        #
        # Unlike cosine-thresholding on stop_embed, this head can capture dataset-specific statistics
        # without forcing the underlying `order_embeds` distribution to align to stop_embed.
        gate_loss = None
        gate_w = _env_float("VIGOR_VARLEN_GATE_W", 0.0)
        gate_pos_w = _env_float("VIGOR_VARLEN_GATE_POS_WEIGHT", 1.0)
        gate_rew_l13 = _env_float("VIGOR_VARLEN_GATE_REWEIGHT_L13", 0.0)
        gate_loss_type = _env_str("VIGOR_VARLEN_GATE_LOSS", "token").strip().lower()  # token | len | both
        gate_mono_w = _env_float("VIGOR_VARLEN_GATE_MONO_W", 0.0)
        self.last_gate_loss = None
        if gate_w > 0:
            try:
                gt_m = order_valid_mask_oracle
                if gt_m is None:
                    stop_tok = _env_str("VIGOR_STOP_TOKEN", "<STOP>")
                    rows = []
                    for i in range(int(B)):
                        row = []
                        for k in range(int(self.order_len)):
                            s = str(step_texts[i][k] or "").strip()
                            valid = (s != "") and (s != stop_tok)
                            row.append(1.0 if valid else 0.0)
                        rows.append(row)
                    gt_m = torch.as_tensor(rows, device=order_embeds.device, dtype=torch.float32)
                else:
                    gt_m = gt_m.to(device=order_embeds.device, dtype=torch.float32)

                # Label: invalid/STOP steps are positive (1).
                y = (gt_m < 0.5).to(dtype=torch.float32)  # [B,O]
                logits = self._varlen_gate_logits(order_embeds)  # [B,O]
                pos_weight = torch.tensor(float(gate_pos_w), device=logits.device, dtype=logits.dtype)
                token_loss = None
                len_loss = None

                if gate_loss_type in {"token", "both"}:
                    per = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")  # [B,O]
                    token_loss = per.mean(dim=1)  # [B]

                if gate_loss_type in {"len", "length", "both"}:
                    # Sequence-level negative log-likelihood of the boundary position L.
                    # Here p_k = sigmoid(logits_k) is P(invalid at step k). GT assumes a single boundary:
                    #   steps < L are valid (y=0), steps >= L invalid (y=1).
                    #
                    # Use logits for numerical stability:
                    #   log p      = -softplus(-logit)
                    #   log(1 - p) = -softplus(logit)
                    logp = -F.softplus(-logits)   # log(sigmoid)
                    log1p = -F.softplus(logits)   # log(1-sigmoid)
                    # L = number of valid steps (0..O)
                    L = (gt_m >= 0.5).float().sum(dim=1).long().clamp(min=0, max=int(self.order_len))

                    # prefix_log1p[:, L] = sum_{k < L} log(1-p_k)
                    prefix = torch.zeros((B, int(self.order_len) + 1), device=logits.device, dtype=logits.dtype)
                    prefix[:, 1:] = log1p.cumsum(dim=1)
                    # suffix_logp[:, L] = sum_{k >= L} log(p_k)
                    suffix = torch.zeros((B, int(self.order_len) + 1), device=logits.device, dtype=logits.dtype)
                    suffix[:, : int(self.order_len)] = logp.flip(1).cumsum(1).flip(1)
                    score = prefix + suffix  # [B, O+1]
                    # NLL of the true L under categorical softmax(score).
                    len_loss = F.cross_entropy(score, L, reduction="none")  # [B]

                # Optional monotonic regularizer: encourage p_{k+1} >= p_k.
                mono = None
                if gate_mono_w > 0 and int(self.order_len) > 1:
                    p = torch.sigmoid(logits)
                    mono = F.relu(p[:, :-1] - p[:, 1:]).mean(dim=1)  # [B]

                # Combine selected losses per sample.
                per_sample = None
                if token_loss is not None and len_loss is not None:
                    per_sample = 0.5 * token_loss + 0.5 * len_loss
                elif token_loss is not None:
                    per_sample = token_loss
                elif len_loss is not None:
                    per_sample = len_loss
                else:
                    per_sample = torch.zeros((B,), device=logits.device, dtype=logits.dtype)
                if mono is not None:
                    per_sample = per_sample + float(gate_mono_w) * mono

                if gate_rew_l13 > 0:
                    L = (gt_m >= 0.5).float().sum(dim=1).long()
                    w = torch.ones_like(per_sample)
                    w = w + float(gate_rew_l13) * ((L == 1) | (L == 3)).float()
                else:
                    w = torch.ones_like(per_sample)

                gate_loss = (per_sample * w).sum() / w.sum().clamp(min=1.0)
                self.last_gate_loss = float(gate_loss.detach().cpu().item())
            except Exception:
                gate_loss = None

        # Optional explicit m_k auxiliary loss: encourage invalid steps to be STOP-like and valid steps to be STOP-unlike.
        #
        # This addresses a common failure mode on NR3D: invalid slots become STOP-like, but valid slots can also
        # drift towards STOP, making first-stop prediction noisy. The auxiliary loss adds a *contrastive* signal
        # that pushes valid steps away from STOP.
        mk_loss = None
        mk_w = _env_float("VIGOR_STOP_MK_W", 0.0)
        mk_type = _env_str("VIGOR_STOP_MK_TYPE", "rank").strip().lower()  # rank | bce
        mk_margin = _env_float("VIGOR_STOP_MK_MARGIN", 0.25)
        mk_rew_l13 = _env_float("VIGOR_STOP_MK_REWEIGHT_L13", 0.0)
        mk_bce_scale = _env_float("VIGOR_STOP_MK_BCE_SCALE", 20.0)
        mk_bce_center = _env_float("VIGOR_STOP_MK_BCE_CENTER", 0.85)
        self.last_mk_loss = None
        self.last_mk_delta = None
        if mk_w > 0:
            try:
                # GT validity mask m_k: prefer dataset-provided oracle (ori_order_len), else fall back to STOP padding.
                gt_m = order_valid_mask
                if gt_m is None:
                    stop_tok = _env_str("VIGOR_STOP_TOKEN", "<STOP>")
                    rows = []
                    for i in range(int(B)):
                        row = []
                        for k in range(int(self.order_len)):
                            s = str(step_texts[i][k] or "").strip()
                            valid = (s != "") and (s != stop_tok)
                            row.append(1.0 if valid else 0.0)
                        rows.append(row)
                    gt_m = torch.as_tensor(rows, device=order_embeds.device, dtype=torch.float32)
                else:
                    gt_m = gt_m.to(device=order_embeds.device, dtype=torch.float32)

                valid = gt_m >= 0.5
                invalid = ~valid

                # Cosine similarity between each slot and STOP: [B,O]
                #
                # IMPORTANT: do NOT detach here, otherwise `mk_loss` becomes a constant term and
                # does not provide gradients to improve chain-length prediction.
                #
                # If you want to restrict the gradients (e.g., only train stop_embed or only
                # train order_embeds), use these env toggles:
                #   - VIGOR_STOP_MK_DETACH_STOP=1
                #   - VIGOR_STOP_MK_DETACH_EMB=1
                mk_detach_stop = _env_flag("VIGOR_STOP_MK_DETACH_STOP", "0")
                mk_detach_emb = _env_flag("VIGOR_STOP_MK_DETACH_EMB", "0")
                stop_vec = self.stop_embed.detach() if mk_detach_stop else self.stop_embed
                emb_vec = order_embeds.detach() if mk_detach_emb else order_embeds
                stop_n = F.normalize(stop_vec.float().view(1, 1, -1), dim=-1)
                emb_n = F.normalize(emb_vec.float(), dim=-1)
                cos = (emb_n * stop_n).sum(dim=-1)

                has_v = valid.any(dim=1)
                has_i = invalid.any(dim=1)
                ok = has_v & has_i
                if ok.any():
                    v_cnt = valid.float().sum(dim=1).clamp(min=1.0)
                    i_cnt = invalid.float().sum(dim=1).clamp(min=1.0)
                    v_mean = (cos * valid.float()).sum(dim=1) / v_cnt
                    i_mean = (cos * invalid.float()).sum(dim=1) / i_cnt
                    delta = i_mean - v_mean  # want large
                    self.last_mk_delta = float(delta[ok].mean().detach().cpu().item())

                    if mk_type in {"bce", "logistic"}:
                        # invalid=1, valid=0 (logits are centered around mk_bce_center).
                        y = invalid.float()
                        logits = mk_bce_scale * (cos - float(mk_bce_center))
                        per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # [B,O]
                        per_sample = per.mean(dim=1)  # [B]
                    else:
                        # Hinge ranking loss on per-sample separation.
                        per_sample = F.relu(float(mk_margin) - delta)  # [B]

                    # Reweight rare chain lengths (L=1 or L=3) to reduce SR3D/L=2 dominance.
                    if mk_rew_l13 > 0:
                        L = valid.float().sum(dim=1).long()
                        w = torch.ones_like(per_sample)
                        w = w + float(mk_rew_l13) * ((L == 1) | (L == 3)).float()
                    else:
                        w = torch.ones_like(per_sample)

                    mk_loss = (per_sample[ok] * w[ok]).sum() / w[ok].sum().clamp(min=1.0)
                    self.last_mk_loss = float(mk_loss.detach().cpu().item())
            except Exception:
                mk_loss = None

        # Optional hard replacement for interpretability: explicitly write STOP embeddings into padded slots.
        if varlen_enabled and (order_valid_mask is not None) and stop_replace:
            try:
                m = order_valid_mask.to(device=order_embeds.device, dtype=order_embeds.dtype).unsqueeze(-1)  # [B,O,1]
                tgt = self.stop_embed.to(device=order_embeds.device, dtype=order_embeds.dtype).view(1, 1, -1)
                order_embeds = order_embeds * m + tgt * (1.0 - m)
            except Exception:
                pass

        # Also expose the STOP-processed embeddings (useful to verify whether replacement is enabled).
        try:
            self.last_order_embeds_post_stop = order_embeds.detach()
        except Exception:
            self.last_order_embeds_post_stop = None

        batch["order_embeds"] = order_embeds
        if varlen_enabled and (order_valid_mask is not None):
            batch["order_valid_mask"] = order_valid_mask
        if lang_embeds is not None:
            # Bypass listener's BERT lang encoder by injecting token-level memory directly.
            batch["lang_embeds"] = lang_embeds

        out = self.listener(batch, epoch)
        extra_loss = 0.0
        if distill_w > 0 and distill is not None:
            extra_loss = extra_loss + distill_w * distill
        if g_w > 0 and distill_g is not None:
            extra_loss = extra_loss + g_w * distill_g
        if distill_w_mp > 0 and distill_mp is not None:
            extra_loss = extra_loss + distill_w_mp * distill_mp
        if g_w_mp > 0 and distill_g_mp is not None:
            extra_loss = extra_loss + g_w_mp * distill_g_mp
        if mp_op_w > 0 and mp_op is not None:
            extra_loss = extra_loss + mp_op_w * mp_op
        if stop_w > 0 and stop_reg is not None:
            extra_loss = extra_loss + stop_w * stop_reg
        if mk_w > 0 and mk_loss is not None:
            extra_loss = extra_loss + mk_w * mk_loss
        if gate_w > 0 and gate_loss is not None:
            extra_loss = extra_loss + gate_w * gate_loss

        if isinstance(extra_loss, torch.Tensor) and extra_loss.numel() == 1:
            if isinstance(out, (list, tuple)) and len(out) >= 1:
                out0 = out[0] + extra_loss.to(device=out[0].device, dtype=out[0].dtype)
                out = (out0,) + tuple(out[1:])
            else:
                out = out + extra_loss.to(device=out.device, dtype=out.dtype)

        if _env_flag("VIGOR_LLM_STEPSLOT_DIAG", "0"):
            with torch.no_grad():
                flat = order_embeds.detach().float().reshape(-1, order_embeds.size(-1))
                if flat.size(0) >= self.order_len:
                    # Cosine between different steps (off-diagonal).
                    flat_n = F.normalize(flat, dim=-1)
                    sim = (flat_n @ flat_n.t()).abs()
                    # Rough aggregate: average off-diagonal for first B steps.
                    sim = sim[: self.order_len, : self.order_len]
                    off = sim[~torch.eye(self.order_len, dtype=torch.bool, device=sim.device)]
                    print(
                        f"[Vigor][llama_stepslot][diag] step_embed_norm(mean,std)=({flat.norm(dim=-1).mean().item():.3f},"
                        f"{flat.norm(dim=-1).std().item():.3f}) offdiag_cos_mean={off.mean().item():.4f}",
                        flush=True,
                    )
                if distill is not None:
                    print(f"[Vigor][llama_stepslot][diag] distill_loss={float(distill.item()):.6f}", flush=True)
                if distill_g is not None:
                    print(f"[Vigor][llama_stepslot][diag] distill_global={float(distill_g.item()):.6f}", flush=True)

        if _env_flag("VIGOR_LLM_STEPSLOT_PROBE", "0"):
            max_batches = _env_int("VIGOR_LLM_STEPSLOT_PROBE_MAX_BATCHES", 1)
            topk = _env_int("VIGOR_LLM_STEPSLOT_PROBE_TOPK", 5)
            try:
                seen = int(getattr(self, "_probe_seen_batches", 0))
            except Exception:
                seen = 0
            if seen < int(max_batches):
                try:
                    setattr(self, "_probe_seen_batches", int(seen) + 1)
                except Exception:
                    pass

                with torch.no_grad():
                    ovm = order_valid_mask
                    if ovm is None:
                        ovm = batch.get("order_valid_mask", None)
                    if ovm is None:
                        ovm = torch.ones((B, self.order_len), device=order_embeds.device, dtype=torch.float32)
                    ovm = ovm.to(device=order_embeds.device, dtype=torch.float32)

                    # STOP alignment on padded slots (representation property).
                    try:
                        tgt = self.stop_embed.to(device=order_embeds.device, dtype=order_embeds.dtype).view(1, 1, -1)
                        mse = (order_embeds - tgt).pow(2).mean(dim=-1)  # [B,O]
                        emb_n = F.normalize(order_embeds.detach().float(), dim=-1)
                        stop_n = F.normalize(self.stop_embed.detach().float().to(device=order_embeds.device).view(1, 1, -1), dim=-1)
                        cos = (emb_n * stop_n).sum(dim=-1)  # [B,O]
                        valid = ovm > 0.5
                        invalid = ~valid
                        v_denom = valid.float().sum().clamp(min=1.0)
                        i_denom = invalid.float().sum().clamp(min=1.0)
                        v_cos = (cos * valid.float()).sum().item() / float(v_denom.item())
                        i_cos = (cos * invalid.float()).sum().item() / float(i_denom.item())
                        v_mse = (mse.float() * valid.float()).sum().item() / float(v_denom.item())
                        i_mse = (mse.float() * invalid.float()).sum().item() / float(i_denom.item())
                    except Exception:
                        v_cos = i_cos = v_mse = i_mse = float("nan")

                    # Listener invariance check: logits should not change on padded steps (varlen gating).
                    inv_delta_max = float("nan")
                    inv_delta_mean = float("nan")
                    try:
                        steps = getattr(self.listener, "last_ref_logits_steps", None)
                        if isinstance(steps, list) and steps:
                            logits = torch.stack([t.to(device=order_embeds.device) for t in steps], dim=1)  # [B,S,N]
                            S = int(logits.size(1))
                            ovm_s = ovm[:, :S]
                            if S >= 2:
                                delta = (logits[:, 1:, :] - logits[:, :-1, :]).abs().max(dim=-1).values  # [B,S-1]
                                invalid_k = (ovm_s[:, 1:] < 0.5)
                                if invalid_k.any():
                                    inv_delta_max = float(delta[invalid_k].max().item())
                                    inv_delta_mean = float(delta[invalid_k].mean().item())
                                else:
                                    inv_delta_max = 0.0
                                    inv_delta_mean = 0.0
                    except Exception:
                        pass

                    ori_len = batch.get("ori_order_len", None)
                    try:
                        if torch.is_tensor(ori_len):
                            ori_len_list = [int(x) for x in ori_len.view(-1).tolist()]
                        elif isinstance(ori_len, (list, tuple)):
                            ori_len_list = [int(x) for x in list(ori_len)]
                        elif ori_len is None:
                            ori_len_list = []
                        else:
                            ori_len_list = [int(ori_len)]
                    except Exception:
                        ori_len_list = []

                    prefix = f"[Vigor][llama_stepslot][probe] batch={seen+1}/{int(max_batches)}"
                    print(
                        f"{prefix} stop_cos(valid={v_cos:.4f}, invalid={i_cos:.4f}) "
                        f"stop_mse(valid={v_mse:.4f}, invalid={i_mse:.4f}) "
                        f"pad_logits_delta(max={inv_delta_max:.6f}, mean={inv_delta_mean:.6f}) "
                        f"ori_order_len(sample0={ori_len_list[0] if ori_len_list else 'n/a'})",
                        flush=True,
                    )

                    # Print one concrete example (sample 0): step texts + target ranks + top-k indices.
                    try:
                        steps = getattr(self.listener, "last_ref_logits_steps", None)
                        if isinstance(steps, list) and steps:
                            logits = torch.stack([t.to(device=order_embeds.device) for t in steps], dim=1)  # [B,S,N]
                            b0 = 0
                            tgt_pos = int(batch["target_pos"][b0].item())
                            ctx = None
                            if torch.is_tensor(batch.get("context_size", None)):
                                ctx = int(batch["context_size"][b0].item())
                            S = int(logits.size(1))
                            S = min(S, self.order_len)
                            texts0 = step_texts[b0] if (isinstance(step_texts, list) and step_texts) else [""] * self.order_len

                            print(f"{prefix} sample0 target_pos={tgt_pos} context_size={ctx if ctx is not None else 'n/a'}", flush=True)
                            for k in range(S):
                                s = logits[b0, k, :].float()
                                if (ctx is not None) and (ctx > 0) and (ctx < int(s.numel())):
                                    s = s.clone()
                                    s[ctx:] = -1e6
                                score_t = float(s[tgt_pos].item()) if 0 <= tgt_pos < int(s.numel()) else float("nan")
                                rank = int((s > score_t).sum().item()) + 1 if score_t == score_t else -1
                                kk = int(min(int(topk), int(s.numel())))
                                top_idx = s.topk(kk).indices.detach().cpu().tolist()
                                v = float(ovm[b0, k].item()) if (ovm is not None) else 1.0
                                print(
                                    f"{prefix} step{k+1} valid={int(v>0.5)} rank={rank} score_t={score_t:.3f} "
                                    f"top{kk}={top_idx} text='{str(texts0[k])}'",
                                    flush=True,
                                )
                    except Exception:
                        pass

        return out
