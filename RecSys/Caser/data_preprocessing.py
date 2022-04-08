import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def encode_user_item_ids(df_all: pd.DataFrame, inference: bool) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    print('encoding...')

    user_id_label_encoder = LabelEncoder()
    item_id_label_encoder = LabelEncoder()

    df_all['user_id'] = user_id_label_encoder.fit_transform(df_all['user_id'].values)
    df_all['item_id'] = item_id_label_encoder.fit_transform(df_all['item_id'].values)

    print('done!')

    if inference is True:
        # encoder.inverse_transform() 으로 decode
        return user_id_label_encoder, item_id_label_encoder


def get_sequence_and_negative(df_all, unique_users, unique_items) -> Tuple[dict, dict]:
    """
    Args:
        df_all (pd.DataFrame): 모든 데이터가 있는 data frame
        unique_users (list): 중복 없이 저장된 모든 users
        unique_items (list): 중복 없이 저장된 모든 items
        train_items (list): min feedback보다 많이 평가된 items
    """

    print('sorted by sequence and make negative smaples per user...')
    dict_pos_sequence = dict()
    dict_negative_samples = dict()


    for user in unique_users:
        user_sequence_items = df_all[df_all['user_id']==user].sort_values(by='timestamp', axis=0)['item_id'].tolist()
        user_negative_items = np.setdiff1d(unique_items, np.unique(user_sequence_items))
        dict_negative_samples[user] = user_negative_items
        dict_pos_sequence[user] = user_sequence_items

    print('doen!')

    return dict_pos_sequence, dict_negative_samples


def trian_test_split(
    dict_pos_sequence,
    num_test,
    unique_users,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Args:
        df_pos_user_sequnce (dict): user가 본 영화를 timestamp 기준으로 정렬한 것이 저장된 dictionary
        num_test (int): 전체 중 user당 test 개수
        unique_users (list): 중복 없이 저장된 모든 user
    """

    print('split data to train and test...')

    dict_train, dict_test = dict(), dict()

    for user in unique_users:
            list_items = dict_pos_sequence[user]

            # train과 test 개수에 따라 
            list_user_train_items = list_items[:-num_test]
            list_user_test_items = list_items[-num_test:]

            dict_train[user] = list_user_train_items
            dict_test[user] = list_user_test_items

    print('done!')
        
    return dict_train, dict_test


def to_sequence(dict_train, dict_test, sequence_length, target_length):
        """
        Transform to sequence form.
        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:
        
        sequences:
           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6]]
        
        targets:
           [[6, 7, 8],
            [7, 8, 9]]
        
        sequence for test (the last 'sequence_length' items of each user's sequence):
        [[5, 6, 7, 8, 9]]
        Parameters

        Args:
            dict_pos_sequence (dict): user가 본 영화를 timestamp 기준으로 정렬한 dictionary
            sequence_length (int): L의 값으로, 참고할 item의 수
            target_length (int): T의 값으로, 예측할 item의 수
        """

        max_sequence_length = sequence_length + target_length

        sequences = list()
        sequences_targets = list()
        sequence_users = list()

        test_sequences = list()
        test_users = list()
        test_sequences_targets = list()
        

        for user in dict_train.keys():
            for seq in _sliding_window(dict_train[user], max_sequence_length):
                sequence_users.append(user)
                sequences.append(seq[:sequence_length])
                sequences_targets.append(seq[-target_length:])
            
            test_users.append(user)
            test_sequences.append(dict_train[user][-sequence_length:])
            test_sequences_targets.append(dict_test[user])


        train_meta_sequences = SequenceData(sequence_users, sequences, sequences_targets)
        test_meta_sequences = SequenceData(test_users, test_sequences, test_sequences_targets)

        return train_meta_sequences, test_meta_sequences


def _sliding_window(list_items, window_size, step_size=1):
    if len(list_items) - window_size >= 0:
        for i in range(len(list_items), 0, -step_size):
            if i - window_size >= 0:
                yield list_items[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(list_items)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(list_items, (num_paddings, 0), 'constant')


class SequenceData():
    def __init__(self, user_ids, sequences, targets=None):
        self.sequence_users = np.array(user_ids, dtype=np.int64)
        self.sequences = np.array(sequences, dtype=np.int64)
        self.L = self.sequences.shape[1]

        self.sequences_targets = None
        self.T = None
        if np.any(targets):
            self.sequences_targets = np.array(targets, dtype=np.int64)
            self.T = self.sequences_targets.shape[1]


def to_sequence_inference(dict_all, sequence_length):
    users = list()
    sequences = list()
    for user in dict_all.keys():
        users.append(user)
        sequences.append(dict_all[user][-sequence_length:])
    
    data_meta = SequenceData(users, sequences)

    return data_meta

