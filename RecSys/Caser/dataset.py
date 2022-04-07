from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset(Dataset):
    def __init__(self, train_meta):
        self.sequences = train_meta.sequences
        self.sequence_users = train_meta.sequence_users
        self.sequences_targets = train_meta.sequences_targets

        # train_meta_sequences = SequenceData(sequence_users, sequences, sequences_targets)
        # test_meta_sequences = SequenceData(test_users, test_sequences)
    
    def __len__(self,):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        user = self.sequence_users[idx]
        sequence = self.sequences[idx]
        sequence_target = self.sequences_targets[idx]

        return user, sequence, sequence_target

class ValidDataset(Dataset):
    def __init__(self, valid_meta):
        self.sequences = valid_meta.sequences
        self.sequence_users = valid_meta.sequence_users


    def __len__(self,):
        return self.sequences.shape[0]


    def __getitem__(self, idx):
        user = self.sequence_users[idx]
        sequence = self.sequences[idx]

        return user, sequence


def get_dataloader(train_meta, valid_meta, batch_size):
    train_dataset = TrainDataset(train_meta)
    valid_dataset = ValidDataset(valid_meta)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader

