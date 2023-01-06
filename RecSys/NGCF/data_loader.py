import os
import random as rd
import numpy as np
import scipy.sparse as sp
from time import time
import datetime

class Data():
    def __init__(self, path_dir, batch_size):
        self.path_dir = path_dir
        self.batch_size = batch_size
        
        self.n_users, self.n_items = 0, 0 # num of user, item
        self.n_train, self.n_test = 0, 0 # num of train, test interaction
        self.neg_pools = {} # neg samples for each user
        self.users = [] # existing users, need for batch sampling

        file_train = os.path.join(self.path_dir, 'train.txt') # uid, item1, item2, ... per line
        file_test = os.path.join(self.path_dir, 'test.txt')
        
        self.train_interactions = {} # pos samples for each train user
        with open(file_train) as f:
            for l in f.readlines(): # readlines(): return list of file lines.
                if len(l) > 0:
                    l = list(map(int, l.strip('\n').split(' ')))
                    uid, items = l[0], l[1:]
                    self.users.append(uid)                    
                    # highest number is the number of users/items
                    self.n_users = max(self.n_users, uid)
                    self.n_items = max(self.n_items, max(items))

                    # store train interactions
                    self.train_interactions[uid] = items
                    # number of train interactions
                    self.n_train += len(items)

        self.test_interactions = {}
        with open(file_test) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = list(map(int, l.strip('\n').split(' ')))
                    uid, items = l[0], l[1:]

                    # highest number is the number of items
                    self.n_items = max(self.n_items, max(items))

                    # store test interactions
                    self.test_interactions[uid] = items
                    # number of test interactions
                    self.n_test += len(items)

        # adjust counters: user_id/item_id starts at 0
        self.n_users += 1
        self.n_items += 1

        self._print_statistics()

        print('Creating interactions matrices R_train, R_test')
        st = time()
        self.R_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        for uid, items in self.train_interactions.items():
            for item in items:
                self.R_train[uid, item] = 1.0
        
        for uid, items in self.test_interactions.items():
            for item in items:
                self.R_test[uid, item] = 1.0

        print(f'Complete. Took {datetime.timedelta(seconds=int(time() - st))}')
            
    def _print_statistics(self):
        print(f'num users={self.n_users}, num items={self.n_items}')
        print(f'num train interactions={self.n_train}, num test interactions={self.n_test}, num total interactions={self.n_train+self.n_test}')
        print(f'sparsity={(self.n_train + self.n_test)/(self.n_users * self.n_items):.5f}')

    # if exist, get adjacency matrix
    def get_adj_mat(self):
        try:
            print("Load adj matrix")
            st = time()
            adj_mat = sp.load_npz(self.path_dir + '/s_adj_mat.npz')
            print(f"Complete, (shape:'{adj_mat.shape}'). Took {datetime.timedelta(seconds=int(time() - st))}")
        except Exception:
            adj_mat = self._create_adj_mat()
            sp.save_npz(self.path_dir + '/s_adj_mat.npz', adj_mat)
        return adj_mat

    def _create_adj_mat(self):

        print('Creating adj matrix for training')
        st = time()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R_train.tolil() # list of lists: row-based linked list

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        print(f'Complete. Took {datetime.timedelta(seconds=int(time() - st))}')

        def normalized_adj_matrix(adj):
            row_degree = np.array(adj.sum(1)).flatten() # shape: (n_users + n_items, )
            degree_inv = np.power(row_degree, -.5) # D^-0.5
            degree_inv[np.isinf(degree_inv)] = 0.
            d_mat_inv = sp.diags(degree_inv)
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv) # csr

            return norm_adj.tocsr()
        
        st = time()
        print('Transforming adjacency-matrix to Normalized-adjacency matrix...')
        ngcf_adj_mat = normalized_adj_matrix(adj_mat)
        print(f'Complete. Took {datetime.timedelta(seconds=int(time() - st))}')
        return ngcf_adj_mat

    # create collections of N items that users never interacted with
    def _negative_pool(self):
        st = time()
        for u in self.train_interactions.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_interactions[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print(f'refresh negative pools. Took {datetime.timedelta(seconds=int(time() - st))}')
    
    # sample data for mini-batches
    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.users, self.batch_size)
        else:
            users = [rd.choice(self.users) for _ in range(self.batch_size)] # allow duplicate users

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_interactions[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_id = pos_items[pos_idx]

                if pos_id not in pos_batch:
                    pos_batch.append(pos_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_interactions[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            if not self.neg_pools: self._negative_pool()
            neg_items = list(set(self.neg_pools[u]) - set(self.train_interactions[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items