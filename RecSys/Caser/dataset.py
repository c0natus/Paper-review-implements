import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CaserDataset(Dataset):
    def __init__(self, meta, inference):
        self.sequences = meta.sequences
        self.sequence_users = meta.sequence_users
        self.sequences_targets = meta.sequences_targets

        self.inference = inference
            

    def __len__(self,):
        return self.sequences.shape[0]


    def __getitem__(self, idx):
        user = self.sequence_users[idx]
        sequence = self.sequences[idx]

        user = torch.tensor(user).to(torch.int64)
        sequence = torch.from_numpy(sequence).long()

        if self.inference is True:
            return user, sequence

        sequence_target = self.sequences_targets[idx]
        sequence_target = torch.from_numpy(sequence_target).long()

        return user, sequence, sequence_target


def get_dataloader(meta, batch_size, inference=False):
    dataset = CaserDataset(meta, inference=inference)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return dataloader

