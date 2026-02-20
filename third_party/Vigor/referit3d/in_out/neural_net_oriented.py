import re
import pathlib
import os.path as osp
import pandas as pd
import numpy as np
from ast import literal_eval

from .vocabulary import build_vocab, Vocabulary
from ..utils import read_lines, unpickle_data
from ..data_generation.nr3d import decode_stimulus_string


def scannet_official_train_val(valid_views=None, verbose=True):
    """
    :param valid_views: None or list like ['00', '01']
    :return:
    """
    pre_fix = osp.split(pathlib.Path(__file__).parent.absolute())[0]
    train_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_train.txt')
    train_split = read_lines(train_split)
    test_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_val.txt')
    test_split = read_lines(test_split)

    if valid_views is not None:
        train_split = [sc for sc in train_split if sc[-2:] in valid_views]
        test_split = [sc for sc in test_split if sc[-2:] in valid_views]

    if verbose:
        print('#train/test scans:', len(train_split), '/', len(test_split))

    scans_split = dict()
    scans_split['train'] = set(train_split)
    scans_split['test'] = set(test_split)
    return scans_split


def objects_counter_percentile(scan_ids, all_scans, prc):
    all_obs_len = list()
    for scan_id in all_scans:
        if scan_id in scan_ids:
            all_obs_len.append(len(all_scans[scan_id].three_d_objects))
    if len(all_obs_len) == 0:
        return 0
    return np.percentile(all_obs_len, prc)


def mean_color(scan_ids, all_scans):
    mean_rgb = np.zeros((1, 3), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points
    return mean_rgb


def load_referential_data(args, referit_csv, scans_split):
    """
    :param args:
    :param referit_csv:
    :param scans_split:
    :return:
    """
    # NOTE: Historically we pass the "*_train_*.csv" and auto-load the matching
    # test csv by swapping "train" -> "test". Some downstream utilities (e.g.,
    # token-length diagnostics) assume there exists at least one training scan.
    #
    # For robustness we also accept passing the "*_test_*.csv" directly.
    # In that case we try to load the matching train csv, and we avoid
    # concatenating the same file twice.
    train_csv = referit_csv
    test_csv = None
    if 'train' in referit_csv:
        test_csv = re.sub(r'_\d+\.\d+', '', referit_csv.replace('train', 'test'))
    elif 'test' in referit_csv:
        test_csv = referit_csv
        train_csv = re.sub(r'_\d+\.\d+', '', referit_csv.replace('test', 'train'))

    dfs = []
    if train_csv and osp.exists(train_csv):
        dfs.append(pd.read_csv(train_csv))
    if test_csv and osp.exists(test_csv) and (test_csv != train_csv):
        dfs.append(pd.read_csv(test_csv))
    if not dfs:
        raise FileNotFoundError(f"Could not load referit3d csv: {referit_csv}")

    referit_data = pd.concat(dfs, ignore_index=True, sort=False)

    if args.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data['mentions_target_class']]
        referit_data.reset_index(drop=True, inplace=True)
        print('Dropping utterances without explicit '
              'mention to the target class {}->{}'.format(n_original, len(referit_data)))

    base_cols = [
        'tokens',
        'instance_type',
        'scan_id',
        'dataset',
        'target_id',
        'utterance',
        'stimulus_id',
        'target_object',
        'anchor_objects',
        'summarized_utterance',
        'referential_order',
        'ref_objects_rel',
    ]
    # Optional multi-target GT set (for M3DRef F1 evaluation).
    if 'target_ids' in referit_data.columns:
        base_cols.append('target_ids')
    referit_data = referit_data[base_cols]
    
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = referit_data.scan_id.apply(lambda x: x in scans_split['train'])
    referit_data['is_train'] = is_train

    # Trim data based on token length
    train_token_lens = referit_data.tokens[is_train].apply(lambda x: len(x))
    if len(train_token_lens) > 0:
        print('{}-th percentile of token length for remaining (training) data'
              ' is: {:.1f}'.format(95, np.percentile(train_token_lens, 95)))
    else:
        # If the caller provided a test-only csv (or split filtering removed all train scans),
        # avoid crashing diagnostics; fall back to reporting over the available data.
        all_token_lens = referit_data.tokens.apply(lambda x: len(x))
        if len(all_token_lens) > 0:
            print('[WARN] No training scans found in referit_data; '
                  'reporting token-length percentile over available data.'
                  ' {}-th percentile is: {:.1f}'.format(95, np.percentile(all_token_lens, 95)))
        else:
            print('[WARN] Empty referit_data after filtering; token-length diagnostics skipped.')
    n_original = len(referit_data)
    referit_data = referit_data[referit_data.tokens.apply(lambda x: len(x) <= args.max_seq_len)]
    referit_data.reset_index(drop=True, inplace=True)
    print('Dropping utterances with more than {} tokens, {}->{}'.format(args.max_seq_len, n_original, len(referit_data)))

    # do this last, so that all the previous actions remain unchanged
    if args.augment_with_sr3d is not None:
        # NOTE: although the flag is called "augment_with_sr3d", in our codebase we
        # treat it as a generic augmentation csv (it just needs to match the same columns).
        # args.s_vs_n_weight (if provided) is interpreted as:
        #   |aug_train| ~= |base_train| * s_vs_n_weight
        # This makes it easy to control the final train mixing ratio.
        print('Adding augmentation dataset.')
        aug_df = pd.read_csv(args.augment_with_sr3d)
        aug_df.tokens = aug_df['tokens'].apply(literal_eval)
        is_train_aug = aug_df.scan_id.apply(lambda x: x in scans_split['train'])
        aug_df['is_train'] = is_train_aug
        aug_df = aug_df[is_train_aug]
        aug_df = aug_df[referit_data.columns]

        base_train_n = int(referit_data['is_train'].sum())
        aug_train_n = int(len(aug_df))
        if args.s_vs_n_weight is not None:
            try:
                w = float(args.s_vs_n_weight)
            except Exception:
                w = None
            if w is not None:
                desired_aug_n = int(round(base_train_n * w))
                if desired_aug_n <= 0:
                    print(f"Skipping augmentation because desired_aug_n={desired_aug_n} (s_vs_n_weight={w}).")
                    aug_df = aug_df.iloc[0:0]
                elif desired_aug_n < aug_train_n:
                    # Subsample without replacement for reproducibility.
                    aug_df = aug_df.sample(n=desired_aug_n, replace=False, random_state=args.random_seed)
                    aug_train_n = int(len(aug_df))
                    print(f"Subsampled augmentation to {aug_train_n} (s_vs_n_weight={w}, base_train={base_train_n}).")
                else:
                    print(f"Using full augmentation set ({aug_train_n}); desired_aug_n={desired_aug_n} (s_vs_n_weight={w}).")

        print('Dataset-size before augmentation:', len(referit_data))
        referit_data = pd.concat([referit_data, aug_df], axis=0)
        referit_data.reset_index(inplace=True, drop=True)
        print('Dataset-size after augmentation:', len(referit_data))

    context_size = referit_data[~referit_data.is_train].stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    print('(mean) Random guessing among target-class test objects {:.4f}'.format( (1 / context_size).mean() ))

    return referit_data


def load_scan_related_data(preprocessed_scannet_file, verbose=True, add_pad=True):
    _, all_scans = unpickle_data(preprocessed_scannet_file)
    if verbose:
        print('Loaded in RAM {} scans'.format(len(all_scans)))

    instance_labels = set()
    for scan in all_scans:
        idx = np.array([o.object_id for o in scan.three_d_objects])
        instance_labels.update([o.instance_label for o in scan.three_d_objects])
        assert np.all(idx == np.arange(len(idx)))  # assert the list of objects-ids -is- the range(n_objects).
                                                # because we use this ordering when we sample objects from a scan.
    all_scans = {scan.scan_id: scan for scan in all_scans}  # place scans in dictionary


    class_to_idx = {}
    i = 0
    for el in sorted(instance_labels):
        class_to_idx[el] = i
        i += 1

    if verbose:
        print('{} instance classes exist in these scans'.format(len(class_to_idx)))

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx['pad'] = len(class_to_idx)

    scans_split = scannet_official_train_val()

    return all_scans, scans_split, class_to_idx


def compute_auxiliary_data(referit_data, all_scans, args):
    """Given a train-split compute useful quantities like mean-rgb, a word-vocabulary.
    :param referit_data: pandas Dataframe, as returned from load_referential_data()
    :param all_scans:
    :param args:
    :return:
    """
    # Vocabulary
    if args.vocab_file:
        vocab = Vocabulary.load(args.vocab_file)
        print(('Using external, provided vocabulary with {} words.'.format(len(vocab))))
    else:
        train_tokens = referit_data[referit_data.is_train]['tokens']
        vocab = build_vocab([x for x in train_tokens], args.min_word_freq)
        print(('Length of vocabulary, with min_word_freq={} is {}'.format(args.min_word_freq, len(vocab))))

    if all_scans is None:
        return vocab

    # Mean RGB for the training
    training_scan_ids = set(referit_data[referit_data['is_train']]['scan_id'])
    print('{} training scans will be used.'.format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans)

    # Percentile of number of objects in the training data
    prc = 90
    obj_cnt = objects_counter_percentile(training_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (training) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))

    # Percentile of number of objects in the testing data
    prc = 99
    testing_scan_ids = set(referit_data[~referit_data['is_train']]['scan_id'])
    obj_cnt = objects_counter_percentile(testing_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (testing) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))
    return mean_rgb, vocab


def trim_scans_per_referit3d_data(referit_data, scans):
    # remove scans not in referit_data
    in_r3d = referit_data.scan_id.unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print('Dropped {} scans to reduce mem-foot-print.'.format(len(to_drop)))
    return scans


##
## Are below necessary? Refactor. Maybe send to a _future_ package
## I think I wrote them to extract the classes of only the training data, but we rejected this idea.
##

# def object_classes_of_scans(scan_ids, all_scans, verbose=False):
#     """ get the object classes (e.g., chair, table...) that the specified scans contain.
#     :param scan_ids: a list of strings
#     :param all_scans: a dictionary holding ScannetScan objects
#     :return: a dictionary mapping all present object classes (string) to a unique int
#     """
#     object_classes = set()
#     for scan_id, scan in all_scans.items():
#         if scan_id in scan_ids:
#             object_classes.update([s.instance_label for s in scan.three_d_objects])
#
#     if verbose:
#         print('{} object classes were found.'.format(len(object_classes)))
#     return object_classes
#
#
# def object_class_to_idx_dictionary(object_classes, add_pad=False):
#     class_to_idx = {m: i for i, m in enumerate(sorted(list(object_classes)))}
#     if add_pad:
#         class_to_idx['pad'] = len(class_to_idx)
#     return class_to_idx
