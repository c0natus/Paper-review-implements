import argparse
import torch
import torch.nn as nn
import os

from utils import set_seed, check_path
from data_preprocessing import *
from dataset import get_dataloader
from caser import Caser
from metric import get_Recall, get_NDCG

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
        output,
        ):
        """
        Args:

        """

        self.epochs = epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.neg_samples = dict_negative_samples
        self.num_neg_samples = num_neg_samples
        self.device = device
        self.output = output

        self.loss_list = list()
        self.recall_list = list()
        self.ndcg_list = list()

        self.model_name = 'Caser'
        self.topK = 10
    
    def fit(self):
        best_ndcg = 0
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)

        self.model.to(device)
        for epoch in range(self.epochs):
            # 시작 시간 기록
            epoch_start.record()

            avg_loss = self._train()
            avg_recall, avg_ndcg = self._metric()
        
            epoch_end.record()
            torch.cuda.synchronize()

            self.loss_list.append(avg_loss)
            self.recall_list.append(avg_recall)
            self.ndcg_list.append(avg_ndcg)


            print(
                f'Epoch[{epoch+1}/{self.epochs}]\ttrain_loss: {avg_loss:.4f}' +
                f'\trecall: {avg_recall:.4f}\tNDCG: {avg_ndcg:.4f} '+
                f'\t훈련시간: {epoch_start.elapsed_time(epoch_end)/1000:.2f} sec'
            )

            if best_ndcg < avg_ndcg:
                best_ndcg = avg_ndcg
                torch.save(self.model.state_dict(), os.path.join(self.output, f'best_NDCG_{self.model_name}.pt'))
                print(f'save ndcg: {best_ndcg:.4f}')


    def _train(self):
        self.model.train()
        size = len(self.train_loader)
        epoch_loss = 0

        for users, sequence, sequence_target in self.train_loader:
            users = users.to(self.device)
            sequence = sequence.to(self.device)
            target_pos = sequence_target.to(self.device)
            target_neg = self._get_neg_smaples(users, self.num_neg_samples).to(self.device)

            input_targets = torch.cat((target_pos, target_neg), dim=-1)
            ground_truth = self._get_GT(target_pos.shape[0], target_pos.shape[1]).to(self.device)

            predict = self.model(users, sequence, input_targets)

            self.optimizer.zero_grad()
            loss = self.criterion(predict, ground_truth)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / size

        return avg_loss


    def _metric(self):
        self.model.eval()
        size = len(self.valid_loader)

        epoch_recall, epoch_ndcg = 0, 0

        with torch.no_grad():
            for users, sequence, sequence_target in self.valid_loader:
                users = users.to(self.device)
                sequence = sequence.to(self.device)
                target_pos = sequence_target.to(self.device)
                all_neg_targets = self._get_neg_smaples(users).to(self.device)

                input_targets = torch.cat((target_pos, all_neg_targets), dim=-1)
                predict = self.model(users, sequence, input_targets, for_pred=True)
                _, indices = torch.topk(predict, dim=0, k=self.topK)
                rank_list = torch.take(input_targets, indices).cpu().numpy()
                target_list = target_pos.squeeze().cpu().numpy()

                epoch_recall += get_Recall(rank_list, target_list)
                epoch_ndcg += get_NDCG(rank_list, target_list)

        avg_hr = epoch_recall / size
        avg_ndcg = epoch_ndcg / size

        return avg_hr, avg_ndcg

    
    def _get_neg_smaples(self, users, num_neg=None):
        neg_items = list()
        users = users.detach().cpu().numpy()
        for user in users:
            if num_neg is None:
                # 모든 negative를 선택한다.
                num_neg = len(self.neg_samples[user])
            items = np.random.choice(self.neg_samples[user], min(len(self.neg_samples), num_neg), replace=False)
            neg_items.append(items)
        
        return torch.from_numpy(np.array(neg_items)).long()

    
    def _get_GT(self, batch_size, num_pos):
        np_pos = np.ones(shape=(batch_size, num_pos), dtype=np.int64)
        np_neg = np.zeros(shape=(batch_size, self.num_neg_samples), dtype=np.int64)

        np_gt = np.concatenate((np_pos, np_neg), axis=-1)

        return torch.from_numpy(np_gt).float()
        

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
    model_parser.add_argument("--L", default=5, type=int)
    model_parser.add_argument("--T", default=3, type=int)

    # hyper args
    hyper_parser = argparse.ArgumentParser()

    hyper_parser.add_argument("--batch_size", default=512, type=int)
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
    encode_user_item_ids(df_all, inference=False)
    #################################################

    # get positive items sorted by timestamp and negavite items per user #
    unique_users, unique_items = df_all['user_id'].unique(), df_all['item_id'].unique()
    dict_user_item, dict_negative_samples = get_sequence_and_negative(df_all, unique_users, unique_items)
    ######################################################################

    # get train valid dataloader #
    dict_train, dict_valid = trian_test_split(dict_user_item, config.num_valid_item, unique_users)
    train_meta, valid_meta = to_sequence(dict_train, dict_valid, model_config.L, model_config.T)
    train_dataloader = get_dataloader(train_meta, hyper.batch_size)
    valid_dataloader = get_dataloader(valid_meta, 1)
    ##############################

    # trainer args init #
    model = Caser(len(unique_users), len(unique_items), model_config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper.learning_rate)
    #####################

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
        device=device,
        output=config.output_dir
        )

    trainer.fit()