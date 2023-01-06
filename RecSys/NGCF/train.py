import os
import argparse
from data_loader import Data
from model import NGCF
from utils import *
from time import time
import datetime

def main(args):
    set_seed(args.seed)

    dir_dataset = os.path.join(args.dir_data, args.dataset)
    dir_output = os.path.join(dir_dataset, 'output')
    if dir_output:
        mkdir(dir_output)

    data_gen = Data(dir_dataset, args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adj_mat = data_gen.get_adj_mat()

    model = NGCF(
        data_gen.n_users,
        data_gen.n_items,
        args,
        adj_mat,
        device,
    )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(dir_output, 'NDCG', patience=5)

    for epoch in range(1, args.epochs):
        print(f'Epoch: [{epoch}/{args.epochs}]')
        st = time()
        loss = train_one_epoch(args, data_gen, model, optimizer)
        print(args.delimiter.join([
            f'  Training.',
            '-',
            f'Took: {datetime.timedelta(seconds=int(time() - st))}',
            f'Loss: {loss:.4f}'
            ]))
        
        st = time()
        with torch.inference_mode():
            recall, ndcg = evaluate(
                    model.u_final_embeddings.detach(), 
                    model.i_final_embeddings.detach(), 
                    data_gen.R_train, 
                    data_gen.R_test, 
                    args.topK,
                    device,
                    )
        print(args.delimiter.join([
            f'  Evaluate.',
            '-',
            f'Took: {datetime.timedelta(seconds=int(time() - st))}',
            f'Recall@{args.topK}: {recall:.4f}',
            f'NDCG@{args.topK}: {ndcg:.4f}'
            ]))
        early_stop(ndcg, model, optimizer, epoch, args)
        if early_stop.stop is True: break


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch NGCF Training", add_help=add_help)

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dir-data", type=str, default='../Data/NGCF', help='Dataset directory path')
    parser.add_argument("--dataset", type=str, default='ml-100k', help='Dataset name')
    parser.add_argument("--delimiter", type=str, default='  ', help='Delimiter')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epoch.')
    parser.add_argument('--layers-embed-dim', type=str, default='[64, 64, 64]', help='Output sizes of every layer')
    parser.add_argument('--reg', type=float, default=1e-5, help='Regularization.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--node-dropout', type=float, default=0.1,
                        help='Drop probability w.r.t. node dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--mess-dropout', type=float, default=0.1,
                        help='Drop probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--topK', type=int, default=20, help='k order of metric evaluation')

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)