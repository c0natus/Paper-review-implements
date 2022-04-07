import argparse
import torch
import torch.nn as nn
import os

from utils import set_seed, check_path
from data_preprocessing import *
from dataset import get_dataloader
from caser import Caser, CaserLoss

class Trainer():
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_dataloader,
        valid_dataloader,
        epochs,
        dict_negative_samples,
        num_neg_samples,
        device,
        ):
        """
        Args:

        """

        self.epochs = epochs
        self.model = model
        self.criterion = criterion
        self.train_loader = train_dataloader



 

if __name__ == '__main__':
    # config args
    config_parser = argparse.ArgumentParser()

    config_parser.add_argument("--data_dir", default='/opt/ml/paper/RecSys/Data/ml-latest-small', type=str)
    config_parser.add_argument("--output_dir", default="output", type=str)
    config_parser.add_argument("--data_file", default="ratings.csv", type=str)
    config_parser.add_argument("--seed", default=42, type=int)
    config_parser.add_argument("--num_valid_item", default=3, type=int)

    # model args
    model_parser = argparse.ArgumentParser()

    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')

    # hyper args
    hyper_parser = argparse.ArgumentParser()

    hyper_parser.add_argument("--batch_size", default=512, type=int)
    hyper_parser.add_argument("--L", default=5, type=int)
    hyper_parser.add_argument("--T", default=3, type=int)
    hyper_parser.add_argument("--learning_rate", default=1e-3, type=float)
    hyper_parser.add_argument("--num_neg_samples", default=3, type=int)
    hyper_parser.add_argument("--epochs", default=50, type=int)

    config = config_parser.parse_args()
    model_config = model_parser.parse_args()
    hyper = hyper_parser.parse_args()

    set_seed(config.seed)
    check_path(config.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Use {device}') 
    
    # read file and encode both user_id and item_id #
    config.data_file_path = os.path.join(config.data_dir, config.data_file)
    df_all = pd.read_csv(config.data_file_path)
    df_all.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}, inplace=True)
    encode_user_item_ids(df_all, inference=True)
    #################################################

    # get positive items sorted by timestamp and negavite items per user #
    unique_users, unique_items = df_all['user_id'].unique(), df_all['item_id'].unique()
    dict_user_item, dict_negative_samples = get_sequence_and_negative(df_all, unique_users, unique_items)
    ######################################################################


    dict_train, dict_valid = trian_test_split(dict_user_item, config.num_valid_item, unique_users)
    train_meta, valid_meta = to_sequence(dict_train, hyper.L, hyper.T)

    train_dataloader, valid_dataloader = get_dataloader(train_meta, valid_meta, hyper.batch_size)

    model = Caser()
    criterion = CaserLoss() # nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)

    torch.cuda.empty_cache() # if necessary
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        epochs=hyper.epochs,
        dict_negative_samples=dict_negative_samples,
        num_neg_samples=hyper.num_neg_samples,
        device=device
        )


     


