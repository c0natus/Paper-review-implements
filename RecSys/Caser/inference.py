import argparse
import torch
import os

from caser import Caser
from dataset import get_dataloader
from utils import set_seed, check_path
from data_preprocessing import *

def inference(model, dataloader, users, negative_samples, topK, user_encoder, item_encoder, dir_output, device):
    model.to(device)
    model.eval()

    # columns: user_id
    df_item_user = pd.DataFrame(columns=user_encoder.inverse_transform([i for i in range(users)]))
    df_item_user.columns.name = 'user'
    df_item_user.index.name = 'order'
    
    with torch.no_grad():
        for user, sequence in dataloader:
            user = user.to(device)
            sequence = sequence.to(device)
            neg_items = torch.from_numpy(negative_samples[user.item()]).to(device)

            predict = model(user, sequence, neg_items, for_pred=True)
            _, indices = torch.topk(predict, dim=0, k=topK)
            rank_list = torch.take(neg_items, indices).cpu().numpy().tolist()

            origin_user = user_encoder.inverse_transform(user.cpu().numpy())[0]
            origin_items = item_encoder.inverse_transform(rank_list)
        
            df_item_user[origin_user] = origin_items
    
    df_prediction = pd.DataFrame(df_item_user.unstack()).reset_index()[['user', 0]]
    df_prediction.rename(columns={0: 'item'}, inplace=True)
    df_prediction.to_csv(os.path.join(dir_output, 'submission_Caser.csv'), index=False)
    return df_prediction




if __name__ == '__main__':
    # config args
    config_parser = argparse.ArgumentParser()

    config_parser.add_argument("--data_dir", default='/opt/ml/paper/RecSys/Data/ml-latest-small', type=str)
    config_parser.add_argument("--output_dir", default="output", type=str)
    config_parser.add_argument("--data_file", default="ratings.csv", type=str)
    config_parser.add_argument("--seed", default=42, type=int)
    config_parser.add_argument("--num_valid_item", default=0, type=int)
    config_parser.add_argument("--topK", default=10, type=int)

    # model args
    model_parser = argparse.ArgumentParser()

    model_parser.add_argument('--d', type=int, default=50)
    model_parser.add_argument('--nv', type=int, default=4)
    model_parser.add_argument('--nh', type=int, default=16)
    model_parser.add_argument('--drop', type=float, default=0.5)
    model_parser.add_argument('--ac_conv', type=str, default='relu')
    model_parser.add_argument('--ac_fc', type=str, default='relu')
    model_parser.add_argument("--L", default=5, type=int)
    model_parser.add_argument("--T", default=3, type=int)


    config = config_parser.parse_args()
    model_config = model_parser.parse_args()

    set_seed(config.seed)
    check_path(config.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Use {device}') 
    
    # read file and encode both user_id and item_id #
    config.data_file_path = os.path.join(config.data_dir, config.data_file)
    df_all = pd.read_csv(config.data_file_path)
    df_all.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}, inplace=True)
    user_encoder, item_encoder = encode_user_item_ids(df_all, inference=True)
    #################################################

    # get positive items sorted by timestamp and negavite items per user #
    unique_users, unique_items = df_all['user_id'].unique(), df_all['item_id'].unique()
    dict_user_item, dict_negative_samples = get_sequence_and_negative(df_all, unique_users, unique_items)
    ######################################################################

    data_meta = to_sequence_inference(dict_user_item, model_config.L)
    dataloader = get_dataloader(data_meta, 1)
    
    model = Caser(len(unique_users), len(unique_items), model_config)
    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_NDCG_Caser.pt')))
    inference(model, dataloader, len(unique_users), dict_negative_samples, config.topK, user_encoder, item_encoder, config.output_dir, device)