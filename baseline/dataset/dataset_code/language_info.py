import os
import ast
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Any
import json
import random
from copy import deepcopy
from pathlib import Path
from .utils import concatenate_texts_with_separator

# SSR3DLLM: offline teacher distillation helpers (optional).
try:
    from utils.teacher_distill import make_teacher_key  # type: ignore
except Exception:  # pragma: no cover - keep baseline usable if SSR3DLLM is absent
    make_teacher_key = None  # type: ignore

def _load_lang_template() -> dict:
    """
    Load language templates used by baseline language-info generation.

    Internal repos historically referenced `models/LLM/lan_template.json` as a
    cwd-relative path. In the GitHub release snapshot, the baseline LLM assets
    live under `baseline/core/models/LLM/`.
    """
    repo_root = Path(__file__).resolve().parents[3]  # release repo root
    candidates = [
        repo_root / "baseline" / "core" / "models" / "LLM" / "lan_template.json",
        repo_root / "models" / "LLM" / "lan_template.json",
    ]
    for p in candidates:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    raise FileNotFoundError(
        "Cannot locate `lan_template.json`. Tried:\n"
        + "\n".join([f"  - {c}" for c in candidates])
    )


lang_template = _load_lang_template()


class lang_info_data():
    def __init__(self, question=None, answer=None, lang_type=None, positives_question=[], inst_ids_question=[], query_ids_question=None,
                 positives_answer=[], inst_ids_answer=[], query_ids_answer=None):

        self.question = question
        self.answer = answer
        self.lang_type = lang_type

        self.positives_question = positives_question
        self.inst_ids_question = inst_ids_question
        self.query_ids_question = query_ids_question

        self.positives_answer = positives_answer
        self.inst_ids_answer = inst_ids_answer
        self.query_ids_answer = query_ids_answer

        self.assert_ListOfList(positives_question)
        self.assert_ListOfList(inst_ids_question)
        self.assert_ListOfList(query_ids_question)
        self.assert_ListOfList(positives_answer)
        self.assert_ListOfList(inst_ids_answer)
        self.assert_ListOfList(query_ids_answer)

        assert self.lang_type.split(':')[0] in ['detection', 'scanrefer', 'm3dref', 'groundedscenecaption', 'scanqa', 'objdesc', 'scenedesc',
                                                'scan2cap', '3dllm', 'alpaca', 'embodieddialog', 'embodiedplan', "globalscenecap", "rel3dref",
                                                "referit3d"]
        # assert k in ['scanrefer', 'm3dref', 'groundedscenecaption', 'scan2cap', 'scanqa', 'objdesc', 'scenedesc', '3dllm', 'alpaca', 'none']
        assert self.lang_type.split(':')[-1] in ['text_only', 'with_grounding']
        assert self.lang_type.endswith('with_grounding') or self.lang_type.endswith(
            'text_only'), f'{self.lang_type} has not withgrounding or text_only'

        self.max_gt_iou = np.nan

    @classmethod
    def from_grounding(cls,
                       raw_text,
                       lang_type,
                       lang_token_inst_id_pair,
                       map_target_to_query,
                       valid_target,
                       support_counting=False, count_instance=True):

        map_num_to_words = lang_template["numbers_to_words"]

        if lang_type.startswith('detection'):
            det = (random.choice(lang_template["detection"]["questions"]),
                   random.choice(lang_template["detection"]["answers_w_o_s"]),
                   random.choice(lang_template["detection"]["answers_wo_o"]),
                   random.choice(lang_template["detection"]["answers_w_o_m"]),
                   random.choice(
                       lang_template["detection"]["counting_problem"]),
                   )
            # TODO: add num to output to improve counting task
            if support_counting:
                gt_inst_ids = np.unique(
                    [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
                probability_of_counting = random.random()
                if probability_of_counting < 0.8:  # counting problem
                    input_text = det[4].format(category=raw_text)
                else:  # det only
                    input_text = det[0].format(category=raw_text)
                if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                    gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                    )
                    if len(valid_target[gt_inst_ids]) > 1:  # multi objects
                        prepare_lan_text = raw_text
                        if count_instance:
                            if probability_of_counting < 0.8:  # counting problem
                                if len(valid_target[gt_inst_ids]) <= 20:
                                    prepare_lan_text = f"{map_num_to_words[str(len(valid_target[gt_inst_ids]))]} {prepare_lan_text}"
                                else:  # use digits
                                    prepare_lan_text = f"{len(valid_target[gt_inst_ids])} {prepare_lan_text}"
                            else:  # det only
                                pass  # Use original text
                        positive = det[3].find("{")
                        positive = (positive, positive+len(prepare_lan_text)+1)
                        output_text = det[3].format(category=prepare_lan_text)
                    else:  # single object
                        positive = det[1].find("{")
                        positive = (positive, positive+len(raw_text))
                        output_text = det[1].format(category=raw_text)
                else:
                    positive = []
                    gt_queries_id = []
                    output_text = det[2].format(category=raw_text)
            else:
                gt_inst_ids = np.unique(
                    [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
                input_text = det[0].format(category=raw_text)
                if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                    gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                    )
                    if len(valid_target[gt_inst_ids]) > 1:  # multi objects
                        prepare_lan_text = raw_text
                        if count_instance:
                            # manually filter: "Several {category}s have been identified in this indoor setting.",
                            if random.random() < 0.5 and not det[3].startswith("Several "):
                                if random.random() < 0.5 and len(valid_target[gt_inst_ids]) <= 20:
                                    prepare_lan_text = f"{map_num_to_words[str(len(valid_target[gt_inst_ids]))]} {prepare_lan_text}"
                                else:  # use digits
                                    prepare_lan_text = f"{len(valid_target[gt_inst_ids])} {prepare_lan_text}"
                            else:
                                pass  # Use original text
                        positive = det[3].find("{")
                        positive = (positive, positive+len(prepare_lan_text)+1)
                        output_text = det[3].format(category=prepare_lan_text)
                    else:  # single object
                        positive = det[1].find("{")
                        positive = (positive, positive+len(raw_text))
                        output_text = det[1].format(category=raw_text)
                else:
                    positive = []
                    gt_queries_id = []
                    output_text = det[2].format(category=raw_text)
        elif lang_type.startswith('scanrefer') or lang_type.startswith('referit3d'):
            grd = (random.choice(lang_template["grounding"]["questions"]),
                   random.choice(lang_template["grounding"]["answers_w_o_s"]),
                   random.choice(lang_template["grounding"]["answers_wo_o"]))
            gt_inst_ids = np.unique(
                [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
            input_text = grd[0].format(grounding_text=raw_text)
            if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                )
                positive = grd[1].find("{")
                positive = (positive, positive+len("object"))
                output_text = grd[1].format(category="object")
            else:
                positive = []
                gt_queries_id = []
                output_text = grd[2].format(category="object")
            obj = cls(
                question=input_text,
                answer=output_text,
                lang_type=lang_type,
                positives_answer=[positive],
                inst_ids_answer=[gt_inst_ids.tolist()],
                query_ids_answer=[gt_queries_id],
            )
            return obj
        elif lang_type.startswith('m3dref'):
            multi_grd = (random.choice(lang_template["multi_grounding"]["questions"]),
                         random.choice(
                             lang_template["multi_grounding"]["answers_w_o_s"]),
                         random.choice(
                             lang_template["multi_grounding"]["answers_wo_o"]),
                         random.choice(lang_template["multi_grounding"]["answers_w_o_m"]))
            gt_inst_ids = np.unique(
                [gt_inst_id for token_bid, gt_inst_id in lang_token_inst_id_pair])
            input_text = multi_grd[0].format(grounding_text=raw_text)
            if len(gt_inst_ids) > 0 and len(valid_target[gt_inst_ids]) > 0:
                gt_queries_id = map_target_to_query[gt_inst_ids][valid_target[gt_inst_ids]].tolist(
                )
                if len(valid_target[gt_inst_ids]) > 1:  # multi objects
                    prepare_lan_text = "object"
                    if count_instance:
                        if random.random() < 0.5:
                            if random.random() < 0.5 and len(valid_target[gt_inst_ids]) <= 20:
                                prepare_lan_text = f"{map_num_to_words[str(len(valid_target[gt_inst_ids]))]} {prepare_lan_text}"
                            else:  # use digits
                                prepare_lan_text = f"{len(valid_target[gt_inst_ids])} {prepare_lan_text}"
                        else:
                            pass  # Use original text
                    positive = multi_grd[3].find("{")
                    positive = (positive, positive+len(prepare_lan_text)+1)
                    output_text = multi_grd[3].format(
                        category=prepare_lan_text)
                else:  # single object
                    positive = multi_grd[1].find("{")
                    positive = (positive, positive+len("object"))
                    output_text = multi_grd[1].format(category="object")
            else:
                positive = []
                gt_queries_id = []
                output_text = multi_grd[2].format(category="object")
            obj = cls(
                question=input_text,
                answer=output_text,
                lang_type=lang_type,
                positives_answer=[positive],
                inst_ids_answer=[gt_inst_ids.tolist()],
                query_ids_answer=[gt_queries_id],
            )
            return obj
        else:
            # Unknown grounding type; let caller decide whether to skip.
            return None

        # Default path (e.g. detection): return the constructed sample.
        return cls(
            question=input_text,
            answer=output_text,
            lang_type=lang_type,
            positives_answer=[positive],
            inst_ids_answer=[gt_inst_ids.tolist()],
            query_ids_answer=[gt_queries_id],
        )

    @classmethod
    def from_instruction_following(cls,
                                   instruction_item,
                                   train_mode=False
                                   ):
        instruction_item = deepcopy(instruction_item)
        if instruction_item['lang_type'].startswith('scanqa'):
            return cls(
                question=instruction_item['question'],
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':text_only'
            )
        elif instruction_item['lang_type'].startswith('objdesc'):
            question = random.choice(lang_template['object_description'])
            return cls(
                question=question,
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':text_only',
                inst_ids_question=[[instruction_item['object_ids'][0][0]]],
                positives_question=[
                    [question.index('object'), question.index('object') + len('object')]]
            )
        elif instruction_item['lang_type'].startswith('scan2cap'):
            question = random.choice(lang_template['scan2cap'])
            return cls(
                question=question,
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':text_only',
                inst_ids_question=[[instruction_item['object_ids'][0][0]]],
                positives_question=[
                    [question.index('object'), question.index('object') + len('object')]],
            )
        elif instruction_item['lang_type'].startswith('scenedesc'):
            question = random.choice(lang_template['scene_description'])
            # flatten inst ids
            unique_object_ids = np.unique(
                [j for i in instruction_item['object_ids'] for j in i]).tolist()
            np.random.shuffle(unique_object_ids)

            if train_mode and np.random.rand() < 0.3:
                lang_type = instruction_item['lang_type'] + ':text_only'
            else:
                lang_type = instruction_item['lang_type'] + ':with_grounding'
            return cls(
                question=question,
                answer=instruction_item['answer'],
                lang_type=lang_type,
                inst_ids_question=[unique_object_ids],
                positives_question=[
                    [question.index('objects'), question.index('objects') + len('objects')]],
                inst_ids_answer=instruction_item['object_ids'],
                positives_answer=instruction_item['all_phrases_positions']
            )
        elif instruction_item['lang_type'].startswith('3dllm'):
            return cls(
                question=instruction_item['question'],
                answer=instruction_item['answers'],
                lang_type=instruction_item['lang_type'] + ':text_only'
            )
        elif instruction_item['lang_type'].startswith('embodieddialog'):
            return cls(
                question=instruction_item['question'],
                answer=instruction_item['answer'],
                lang_type=instruction_item['lang_type'] + ':with_grounding',
                inst_ids_question=instruction_item['object_ids_question'],
                inst_ids_answer=instruction_item['object_ids'],
                positives_question=instruction_item['all_phrases_positions_question'],
                positives_answer=instruction_item['all_phrases_positions'],
            )
        elif instruction_item['lang_type'].startswith('embodiedplan'):
            plan_question = instruction_item['question']
            plan_answer = instruction_item['answer']
            plan_question = f"{random.choice(lang_template['planning']['requires']).format(a_high_level_task=str(plan_question).lower())}"
            plan_prefix = f"{random.choice(lang_template['planning']['plan_start'])}\n"
            plan_answer = plan_prefix+plan_answer + \
                f"{random.choice(lang_template['planning']['plan_complete'])}"

            return cls(
                question=plan_question,
                answer=plan_answer,
                lang_type=instruction_item['lang_type'] + ':with_grounding',
                inst_ids_answer=instruction_item['object_ids'],
                positives_answer=[[positive[0]+len(plan_prefix), positive[1]+len(
                    plan_prefix)] for positive in instruction_item['all_phrases_positions']],
            )
        elif instruction_item['lang_type'].startswith('alpaca'):
            return cls(
                question=instruction_item["instruction"] +
                instruction_item["input"],
                answer=instruction_item['output'],
                lang_type=instruction_item['lang_type'] + ":text_only",
            )
        elif instruction_item['lang_type'].startswith('globalscenecap'):
            return cls(
                question=random.choice(lang_template["globalscenecap"]),
                answer=instruction_item["caption"],
                lang_type=instruction_item['lang_type'] + ':text_only',
                inst_ids_answer=instruction_item["object_ids_answer"],
                positives_answer=instruction_item["all_phrases_positions_answer"],
            )
        else:
            raise NotImplementedError

    @classmethod
    def from_relation_sample(cls, rel_item):
        """
        """
        scene_id = rel_item.get("scene_id")
        question = rel_item.get("question")
        rel_text = rel_item.get("final_text_for_training") or rel_item.get("relation_text") or ""
        if not question:
            target_phrase = rel_item.get("target_object_phrase", "object")
            anchor_phrase = rel_item.get("anchor_object_phrase", "other objects")
            question = f"What is the spatial relation between {target_phrase} and {anchor_phrase}?"
        # SSR3DLLM: explicit geometry trigger token for relation reasoning.
        # Downstream geometry heads only train/eval on rel3dref samples that
        # are explicitly marked as needing geometry (either via "<geom>" token
        # or a boolean flag). We keep both for robustness.
        if "<geom>" not in str(question):
            question = f"<geom> {question}"
        # Optionally: supervise the LLM to generate explicit step tokens (teacher forcing).
        # This is used for the "LLM-driven chain" experiments, where the model generates:
        #   door <step1> table <step2> chair <step3> chair <step4>
        # and downstream geometry heads read the hidden states at <stepK>.
        #
        # We intentionally put step text BEFORE <stepK>:
        # - For causal decoders, <stepK> cannot attend to future tokens.
        # - This makes the <stepK> hidden state encode the step semantics.
        use_step_answer = os.environ.get("SSR3DLLM_REL3D_OUTPUT_STEPS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        answer_steps = rel_item.get("answer_steps", None)
        if use_step_answer and isinstance(answer_steps, str) and answer_steps.strip():
            answer = answer_steps.strip()
        else:
            answer = rel_text

        target_gt_id = rel_item.get("target_object_gt_id", None)
        inst_ids_answer = []
        positives_answer = []
        if isinstance(target_gt_id, int):
            inst_ids_answer = [[target_gt_id]]
            positives_answer = [[]]

        inst_ids_question = []
        positives_question = []
        anchor_id_source = rel_item.get("anchor_id_source", "")
        anchor_gt_ids = rel_item.get("anchor_object_gt_ids", None)
        if not isinstance(anchor_gt_ids, list):
            anchor_gt_id = rel_item.get("anchor_object_gt_id", None)
            anchor_gt_ids = [anchor_gt_id] if isinstance(anchor_gt_id, int) else []

        if anchor_id_source == "direct_annotation" and anchor_gt_ids:
            inst_ids_question = [[int(a)] for a in anchor_gt_ids if isinstance(a, int)]
            positives_question = [[] for _ in inst_ids_question]

        rel_type = rel_item.get("relation_type", "generic")
        lang_type = f"rel3dref:{rel_type}:with_grounding"

        obj = cls(
            question=question,
            answer=answer,
            lang_type=lang_type,
            inst_ids_question=inst_ids_question,
            positives_question=positives_question,
            inst_ids_answer=inst_ids_answer,
            positives_answer=positives_answer,
        )
        # Mark as geometry-triggered even if the question is later truncated/edited.
        obj.use_geom_trigger = True
        source = rel_item.get("source_dataset", None)
        if source is not None:
            obj.relation_source = source

        # SSR3DLLM: store minimal metadata for offline teacher-logits distillation.
        # - scene_id / target_object_gt_id are used to build stable sample keys.
        # - distill_text is chosen to be the "utterance-like" text used by teachers.
        obj.rel_scene_id = scene_id
        obj.rel_target_object_gt_id = target_gt_id if isinstance(target_gt_id, int) else None
        distill_text = (
            rel_item.get("distill_text")
            or rel_item.get("utterance")
            or answer
            or question
            or ""
        )
        obj.rel_distill_text = distill_text
        if (
            make_teacher_key is not None
            and isinstance(source, str)
            and isinstance(obj.rel_scene_id, str)
            and isinstance(obj.rel_target_object_gt_id, int)
        ):
            try:
                obj.teacher_key = make_teacher_key(
                    teacher_name=source,
                    scene_id=obj.rel_scene_id,
                    target_gt_id=obj.rel_target_object_gt_id,
                    text=str(distill_text),
                )
            except Exception:
                obj.teacher_key = None

        # SSR3DLLM: carry Vigor-style referential order (step-pointer supervision) when available.
        # This is used when we route rel3dref samples to a pretrained mask3d-vigor listener.
        order_raw = rel_item.get("referential_order", None)
        order_list: List[str] = []
        if isinstance(order_raw, list):
            order_list = [str(x).strip().strip("*").strip() for x in order_raw]
            order_list = [x for x in order_list if x]
        elif isinstance(order_raw, str):
            s = order_raw.strip()
            if s:
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, list):
                        order_list = [str(x).strip().strip("*").strip() for x in v]
                        order_list = [x for x in order_list if x]
                except Exception:
                    order_list = []
        if order_list:
            obj.rel_referential_order = order_list
            # Oracle chain length for VarLen-STOP masking (used by step-slot + Vigor listener).
            try:
                obj.ori_order_len = int(len(order_list))
            except Exception:
                pass

        use_geom_trigger = os.environ.get(
            "SSR3DLLM_USE_GEOM_TRIGGER_TOKEN", "1"
        ).lower() in {"1", "true", "yes", "on"}
        if use_geom_trigger:
            geom_token = "<geom>"
            if obj.question is None:
                obj.question = geom_token
            else:
                if geom_token not in obj.question:
                    obj.question = obj.question + " " + geom_token
            obj.use_geom_trigger = True

        return obj

    def append_prompt_postfix(self):
        # short answer prompt
        if "scanqa" in self.lang_type:
            self.question = self.question + " Please answer with a single word or phrase."

        # with grounding prompt
        if 'with_grounding' in self.lang_type:
            self.question = self.question + ' (with grounding)'

    def set_batch_idx(self, batch_idx):
        self.batch_idx = batch_idx

    def set_max_gt_iou(self, max_gt_iou):
        if 'scan2cap' in self.lang_type or 'objdesc' in self.lang_type:
            assert len(self.inst_ids_question) == 1 and len(
                self.inst_ids_question[0]) == 1
            self.max_gt_iou = max_gt_iou[self.inst_ids_question[0][0]].item()

    def set_context_features(self, query_hidden_feature, query_normalized_embed):
        self.query_hidden_feature = query_hidden_feature
        self.query_normalized_embed = query_normalized_embed

    def assert_ListOfList(self, x):
        if not x:
            return
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], (list, tuple)):
                return

        raise AssertionError('Assertion Error ListofList')

    def remap_inst_ids(self, mapping):
        def inplace_replace_insts(rawtext_posinsts_tmp, instance_mapping_tmp):
            # TODO how to deal with the empty instance ids
            for i, posinsts in enumerate(rawtext_posinsts_tmp):
                for j in range(len(posinsts)):
                    rawtext_posinsts_tmp[i][j] = instance_mapping_tmp[rawtext_posinsts_tmp[i][j]]
        inplace_replace_insts(self.inst_ids_question, mapping)
        inplace_replace_insts(self.inst_ids_answer, mapping)

    def __str__(self):
        return (f'lang_info_data >>> \n') +\
            (f'question: {self.question}\n') +\
            (f'answer: {self.answer}\n') +\
            (f'lang_type: {self.lang_type}\n') +\
            (f'positives_question: {self.positives_question}\n') + \
            (f'inst_ids_question: {self.inst_ids_question}\n') + \
            (f'query_ids_question: {self.query_ids_question}\n') + \
            (f'positives_answer: {self.positives_answer}\n') + \
            (f'inst_ids_answer: {self.inst_ids_answer}\n') + \
            (f'query_ids_answer: {self.query_ids_answer}\n')


class grounding_data:
    def __init__(self):
        self.texts = []
        self.types = []
        self.positives = []
        self.gt_insts = []
        # NOTE: For ScanRefer/M3DRef teacher-forced grounding_steps lookup we need the
        # dataset-defined target instance id (original ScanNet instance id, BEFORE
        # remap_inst_ids() turns instance ids into contiguous indices).
        #
        # By convention (and consistent with tools/check_grounding_steps_coverage.py),
        # target_gt_id is `object_ids[0][0]`.
        self.target_gt_ids: List[Optional[int]] = []

        self.concat_texts, self.concat_positives, self.concat_gt_insts, self.concat_types = [], [], [], []
        # Per-raw-text target ids aligned with concat_* (raw text) axis.
        self.concat_target_gt_ids: List[Optional[int]] = []

    def add_detection(self, class_label, gt_insts):
        self.texts.append(class_label + '.')
        self.gt_insts.append([gt_insts])
        self.positives.append([[0, len(class_label)]])
        self.types.append('detection:with_grounding')
        self.target_gt_ids.append(None)

    def add_grounding(self, grounding_text, gt_insts, positives, grounding_type):
        self.texts.append(deepcopy(grounding_text))
        self.gt_insts.append(deepcopy(gt_insts))
        self.positives.append(deepcopy(positives))
        self.types.append(deepcopy(grounding_type)+':with_grounding')
        tgt: Optional[int] = None
        try:
            if isinstance(gt_insts, (list, tuple)):
                # Prefer the first non-empty instance list as the dataset-defined target id.
                # Some language sources may include degenerate entries with empty object_ids.
                for inst_list in gt_insts:
                    if (
                        isinstance(inst_list, (list, tuple))
                        and len(inst_list) > 0
                        and inst_list[0] is not None
                    ):
                        tgt = int(inst_list[0])
                        break
        except Exception:
            tgt = None
        self.target_gt_ids.append(tgt)

    def shuffle_grounding(self):
        # TODO (random all indices)

        # they are separated with  '. ' so it's okay without shuffling
        if len(self.texts) == 0:
            return

        random_text_indices = np.arange(
            len([typ for typ in self.types if not str(typ).startswith('detection')]),
            dtype=np.int64,
        )
        np.random.shuffle(random_text_indices)

        self.texts = np.asarray(self.texts, dtype=object)
        self.types = np.asarray(self.types, dtype=object)
        self.positives = np.asarray(self.positives, dtype=object)
        self.gt_insts = np.asarray(self.gt_insts, dtype=object)
        self.target_gt_ids = np.asarray(self.target_gt_ids, dtype=object)

        det_count = int(len(self.texts) - len(random_text_indices))
        if det_count < 0:
            det_count = 0

        # Keep detection items in front (unshuffled), shuffle the remaining items.
        withdet_indices = np.asarray(
            np.arange(det_count, dtype=np.int64).tolist()
            + (det_count + random_text_indices).astype(np.int64).tolist(),
            dtype=np.int64,
        )
        if withdet_indices.size == 0:
            return
        self.texts = self.texts[withdet_indices].tolist()
        self.types = self.types[withdet_indices].tolist()
        self.positives = self.positives[withdet_indices].tolist()
        self.gt_insts = self.gt_insts[withdet_indices].tolist()
        self.target_gt_ids = self.target_gt_ids[withdet_indices].tolist()

    def concat_multi_grounding(self, tokenizer, max_batch_tokens, max_tokens, num_concat_texts):
        self.concat_texts, self.concat_positives, self.concat_gt_insts, self.concat_types = concatenate_texts_with_separator(
            tokenizer, self.texts, max_batch_tokens, max_tokens=max_tokens,
            num_concat_texts=num_concat_texts, raw_texts_poschars=self.positives, raw_texts_posinsts=self.gt_insts,
            raw_texts_type=self.types, shuffle=False, text_separator='. ', concat=True)
        # Keep per-raw-text target ids aligned with concat_gt_insts/concat_types.
        try:
            n_raw = int(len(self.concat_types))
        except Exception:
            n_raw = int(len(self.target_gt_ids))
        self.concat_target_gt_ids = deepcopy(self.target_gt_ids[:n_raw])

    def remap_inst_ids(self, mapping):
        self.concat_texts = np.stack(self.concat_texts)

        for gt_insts in self.concat_gt_insts:
            # if gt_insts == -1: continue
            for i, posinsts in enumerate(gt_insts):
                if isinstance(posinsts, (list, np.ndarray)):
                    for j in range(len(posinsts)):
                        gt_insts[i][j] = mapping[gt_insts[i][j]]
                else:
                    assert False
                    gt_insts[i] = mapping[posinsts]

    def compute_positive_maps(self, tokenizer):
        pass
