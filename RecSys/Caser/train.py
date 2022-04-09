import argparse
import torch
import torch.nn as nn
import os

from utils import set_seed, check_path, EarlyStopping
from data_preprocessing import *
from dataset import get_dataloader
from caser import Caser
from metric import get_Recall, get_NDCG

import matplotlib.pyplot as plt
import seaborn as sns

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
        save_metric,
        topK,
        device,
        output,
        save_file_name,
        patience,
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
        
        self.early_stopping = EarlyStopping(
            checkpoint_path=os.path.join(output, save_file_name), 
            patience=patience,
            save_metric=save_metric
            )
        self.save_metric = save_metric
        self.loss_list = list()
        self.recall_list = list()
        self.ndcg_list = list()

        self.topK = topK
    
    def fit(self):
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)

        self.model.to(device)
        print('start training...')
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

            if self.save_metric == 'loss': score = avg_loss
            elif self.save_metric == 'ndcg': score = avg_ndcg
            else: score = avg_recall

            self.early_stopping(score, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        
        print('finish training!')


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

    
    def _get_neg_smaples(self, users, num_neg=1e9):
        neg_items = list()
        users = users.detach().cpu().numpy()
        for user in users:
            items = np.random.choice(self.neg_samples[user], min(len(self.neg_samples[user]), num_neg), replace=False)
            neg_items.append(items)
        
        return torch.from_numpy(np.array(neg_items)).long()

    
    def _get_GT(self, batch_size, num_pos):
        np_pos = np.ones(shape=(batch_size, num_pos), dtype=np.int64)
        np_neg = np.zeros(shape=(batch_size, self.num_neg_samples), dtype=np.int64)

        np_gt = np.concatenate((np_pos, np_neg), axis=-1)

        return torch.from_numpy(np_gt).float()

def plot_loss(epochs, all_loss, model_name, dir_output, loss_name=['Loss', 'Recall', 'NDCG']):
    fig, axes = plt.subplots(1, 3, figsize=(30, 7))
    fig.suptitle(model_name, fontsize=30)
    x_list = [i for i in range(1, epochs+1)]
    
    for i in range(3):
        sns.lineplot(
            x=x_list, y=all_loss[i],
            ax = axes[i]
        )

        axes[i].set_ylabel(loss_name[i])
        axes[i].set_xlabel('Epochs')
    
    plt.show()
    plt.savefig(os.path.join(dir_output, f'{model_name}.png'), dpi=300)
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # config args
    parser.add_argument("--data_dir", default='/opt/ml/paper/RecSys/Data/ml-latest-small', type=str)
    parser.add_argument("--data_file", default="ratings.csv", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_valid_item", default=3, type=int)
    parser.add_argument("--topK", default=10, type=int)

    # model args
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--nh', type=int, default=16)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='relu')
    parser.add_argument("--L", default=5, type=int)
    parser.add_argument("--T", default=3, type=int)

    # hyper args
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--num_neg_samples", default=3, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument('--l2', default=1e-6, type=float)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--save_metric', default='recall', type=str)
    
    config = parser.parse_args()

    config.save_metric = config.save_metric.lower()
    assert config.save_metric in ['ndcg', 'recall', 'loss'], "chooes metric among ndcg, recall and loss"
    config.save_file_name = f"best_{config.save_metric}_Caser.pt"

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
    encode_user_item_ids(df_all, inference=False)
    #################################################

    # get positive items sorted by timestamp and negavite items per user #
    unique_users, unique_items = df_all['user_id'].unique(), df_all['item_id'].unique()
    dict_user_item, dict_negative_samples = get_sequence_and_negative(df_all, unique_users, unique_items)
    ######################################################################

    # get train valid dataloader #
    dict_train, dict_valid = trian_test_split(dict_user_item, config.num_valid_item, unique_users)
    train_meta, valid_meta = to_sequence(dict_train, dict_valid, config.L, config.T)
    train_dataloader = get_dataloader(train_meta, config.batch_size)
    valid_dataloader = get_dataloader(valid_meta, 1)
    ##############################

    # trainer args init #
    model = Caser(len(unique_users), len(unique_items), config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.l2)
    #####################

    torch.cuda.empty_cache() # if necessary
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        epochs=config.epochs,
        dict_negative_samples=dict_negative_samples,
        num_neg_samples=config.num_neg_samples,
        save_metric=config.save_metric,
        topK=config.topK,
        device=device,
        output=config.output_dir,
        save_file_name=config.save_file_name,
        patience=config.patience
        )

    trainer.fit()
    all_loss = [trainer.loss_list, trainer.recall_list, trainer.ndcg_list]
    plot_loss(len(trainer.loss_list), all_loss, 'Caser', config.output_dir)