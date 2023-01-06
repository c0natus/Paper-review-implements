import os
import errno
import numpy as np
import random
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, dir_output, metric: str, patience: int=7, delta: int=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.dir_output = dir_output
        self.save_metric = metric
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def _update(self, score):
        if score > self.best_score + self.delta:
            return True
        return False

    def __call__(self, score, model, optimizer, epoch, args):

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, args)
        elif self._update(score):
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, args)
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True

    def _save_checkpoint(self, model, optimizer, epoch, args):
        """Saves model when the performance is better."""
        print(f"  Better performance. Saving model {self.save_metric}: {self.best_score:.4f} ...")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        torch.save(checkpoint, self.dir_output + '/checkpoint.pth') 


def set_seed(seed: int):
    # https://hoya012.github.io/blog/reproducible_pytorch/
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(args, data_gen, model, optimizer):
    model.train()
    n_batch = data_gen.n_train // args.batch_size + 1
    running_loss = 0
    for _ in range(n_batch):
        u, i, j = data_gen.sample()
        
        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss


def evaluate(user_embed, item_embed, R_train, R_test, topK, device):
    """
    user_embed: User embeddings
    item_embed: Item embeddings
    R_train: Sparse matrix with the training interactions
    R_test: Sparse matrix with the testing interactions
    topK : kth-order for metrics
    
    Returns:
        Dictionary with lists correponding to the metrics at order k for k in Ks
    """

    user_embed_split = split_matrix(user_embed)
    R_train_split = split_matrix(R_train)
    R_test_split = split_matrix(R_test)

    recall_list, ndcg_list = [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(user_embed_split, R_train_split, R_test_split):

        scores = torch.mm(ue_f, item_embed.t())

        test_items = torch.from_numpy(te_f.todense()).float().to(device)
        non_train_items = torch.from_numpy(1-(tr_f.todense())).float().to(device)
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=topK)
        
        pred_items = torch.zeros_like(scores).float()
        pred_items.scatter_(dim=1, index=test_indices, src=torch.ones_like(test_indices).float().to(device))

        TP = (test_items * pred_items).sum(1)
        rec = TP/test_items.sum(1)

        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, topK, device)

        recall_list.append(rec)
        ndcg_list.append(ndcg)

    return torch.cat(recall_list).mean(), torch.cat(ndcg_list).mean()


def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)

    Arguments:
        X: matrix to be split       # X.shape[0]: n_users
        n_folds: number of folds

    Returns:
        splits: split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits


def compute_ndcg_k(pred_items, test_items, test_indices, k, device):
    """
    Compute NDCG@k
    
    Arguments:
    ---------
    pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
    test_items: binary tensor with 1s in locations corresponding to the real test interactions
    test_indices: tensor with the location of the top-k predicted items
    k: k'th-order 

    Returns:
    -------
    NDCG@k
    """
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k+2))).float().to(device)
    
    dcg = (r[:, :k]/f).sum(1)                                               
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k]/f).sum(1)   
    ndcg = dcg/dcg_max                                                     
    
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise