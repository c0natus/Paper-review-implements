import argparse
import torch
import os

from caser import Caser
from dataset import get_dataloader
from utils import set_seed, check_path
from data_preprocessing import *

def inference(model, dataloader, users, negative_samples, topK, user_encoder, item_encoder, dir_output, device, save_file_name):
    model.to(device)
    model.eval()

    # columns: user_id
    df_item_user = pd.DataFrame(columns=user_encoder.inverse_transform([i for i in range(users)]))
    df_item_user.columns.name = 'user'
    df_item_user.index.name = 'order'

    print('start inferencing...')    
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
    df_prediction.to_csv(os.path.join(dir_output, save_file_name), index=False)

    print('done!')




if __name__ == '__main__':
    # config args
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='/opt/ml/paper/RecSys/Data/ml-latest-small', type=str)
    parser.add_argument("--data_file", default="ratings.csv", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--topK", default=10, type=int)
    parser.add_argument('--save_metric', default='ndcg', type=str)

    # model args
    parser.add_argument('--d', default=50, type=int)
    parser.add_argument('--nv', default=4, type=int)
    parser.add_argument('--nh', default=16, type=int)
    parser.add_argument('--drop', default=0.5, type=float)
    parser.add_argument('--ac_conv', default='relu', type=str)
    parser.add_argument('--ac_fc', default='relu', type=str)
    parser.add_argument("--L", default=5, type=int)
    parser.add_argument("--T", default=3, type=int)


    config = parser.parse_args()

    config.save_metric = config.save_metric.lower()
    assert config.save_metric in ['ndcg', 'recall', 'loss'], "chooes metric among ndcg, recall and loss"
    config.load_file_name = f"best_{config.save_metric}_Caser.pt"
    config.save_file_name = f"submission_Caser_{config.save_metric}.csv"

    set_seed(config.seed)
    check_path(config.output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Use {device}') 
    
    # read file and encode both user_id and item_id #
    config.data_file_path = os.path.join(config.data_dir, config.data_file)
    df_all = pd.read_csv(config.data_file_path)
    if 'rating' in df_all.columns.values: df_all = df_all.drop('rating', axis=1)
    column_list = df_all.columns.values
    df_all.rename(columns={column_list[0]: 'user_id', column_list[1]: 'item_id', column_list[2]: 'timestamp'}, inplace=True)
    user_encoder, item_encoder = encode_user_item_ids(df_all, inference=True)
    #################################################

    # get positive items sorted by timestamp and negavite items per user #
    unique_users, unique_items = df_all['user_id'].unique(), df_all['item_id'].unique()
    dict_user_item, dict_negative_samples = get_sequence_and_negative(df_all, unique_users, unique_items)
    ######################################################################

    data_meta = to_sequence_inference(dict_user_item, config.L)
    dataloader = get_dataloader(data_meta, 1, inference=True)
    
    model = Caser(len(unique_users), len(unique_items), config)
    model.load_state_dict(torch.load(os.path.join(config.output_dir, config.load_file_name)))
    inference(
        model=model, 
        dataloader=dataloader, 
        users=len(unique_users), 
        negative_samples=dict_negative_samples, 
        topK=config.topK, 
        user_encoder=user_encoder, 
        item_encoder=item_encoder, 
        dir_output=config.output_dir, 
        device=device,
        save_file_name=config.save_file_name
        )