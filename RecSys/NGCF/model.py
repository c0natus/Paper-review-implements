import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, args, adj_mat, device):
        super(NGCF, self).__init__()

        self.device = device
        self.reg = args.reg
        self.n_users = n_users
        self.n_items = n_items
        self.l_mat = adj_mat
        self.l_plus_i_mat = adj_mat + sp.eye(adj_mat.shape[0])
        self.layers_embed_size = eval(args.layers_embed_dim)
        self.n_layers = len(self.layers_embed_size)
        self.node_dropout = args.node_dropout
        self.mess_dropout = args.mess_dropout

        # Create Matrix 'L+I', PyTorch sparse tensor of SP adjacency_mtx
        self.L = self._convert_sp_mat_to_sp_tensor(self.l_mat)
        self.L_plus_I = self._convert_sp_mat_to_sp_tensor(self.l_plus_i_mat)

        self.user_embeddings = nn.Embedding(self.n_users, self.layers_embed_size[0])
        self.item_embeddings = nn.Embedding(self.n_items, self.layers_embed_size[0])

        self.w1_dict = nn.ModuleDict()
        self.w2_dict = nn.ModuleDict()

        for idx in range(self.n_layers-1):
            self.w1_dict[f'layer_{idx}'] = nn.Linear(self.layers_embed_size[idx], self.layers_embed_size[idx+1], bias=True)
            self.w2_dict[f'layer_{idx}'] = nn.Linear(self.layers_embed_size[idx], self.layers_embed_size[idx+1], bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
    

    # convert sparse matrix into sparse PyTorch tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix(csr)
        """
        coo = X.tocoo().astype(np.float32) # csr to coo
        i = torch.LongTensor(np.mat([coo.row, coo.col])) # shape: (n_users + n_items, n_users + n_items)
        v = torch.FloatTensor(coo.data) # len(coo.data) = train_interactions * 2
        # res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        res = torch.sparse_coo_tensor(
            indices=i,
            values=v,
            size=coo.shape,
            device=self.device,
        )
        return res

    # apply node_dropout
    def _droupout_sparse(self, X):
        """
        Drop individual locations in X
        
        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """

        # _nnz(): the nubmer of non zero elements
        # torch.rand(size): return random numbers from a uniform distribution on the interval [0, 1)
        # floor(): Returns a new tensor with the floor of the elements of input, 
        #   floor: the greatest integer less than or equal to input
        # 1. generate [0, 1) and add node_dropout prob. 
        # 2. make elements of input 0 or 1
        # 3. make elements of input False of True
        node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(self.device)

        # coalesce(): remove duplicate coordinates
        #   the value at that coordinate is the sum of all duplicate value entries.
        #   need for indices() method, even if is_coalesced() is False
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:,node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        # X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)
        X_dropout = torch.sparse_coo_tensor(
            indices=i,
            values=v,
            size=X.shape,
            device=self.device,
        )

        return  X_dropout.mul(1/(1-self.node_dropout)) # why need..?? to keep normalize??

    def forward(self, u: list, p: list, n: list):
        """
        u = user
        p = positive item (user interacted with item)
        n = negative item (user did not interact with item)

        len(u) = len(p) = len(n) = batch size
        """
        # apply node droppout
        L_plus_I_hat = self._droupout_sparse(self.L_plus_I) if self.node_dropout > 0 else self.L_plus_I
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L

        # paper equ. (1)
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0) # shape = (n_users + n_items, layers_embed_size[0])
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers-1):
            # paper equ. (7): message construction + message aggregation

            # paper equ. (3): message construction in matrix form
            L_plus_I_embed = torch.sparse.mm(L_plus_I_hat, ego_embeddings) # (L+I)E
            side_embed = self.w1_dict[f'layer_{k}'](L_plus_I_embed)

            L_embed = torch.mul(torch.sparse.mm(L_hat, ego_embeddings), ego_embeddings) # LE
            interactions_embed = self.w2_dict[f'layer_{k}'](L_embed)

            # paper equ. (4): message aggregation in matrix form
            ego_embeddings = F.leaky_relu(side_embed + interactions_embed)

            # message dropout
            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            # L2 normalize
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            # parper equ. (9): concat outputs
            all_embeddings.append(norm_embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=1)

        # back to user/item dim
        u_final_embeddings, i_final_embeddings = all_embeddings.split([self.n_users, self.n_items], dim=0)

        self.u_final_embeddings = u_final_embeddings
        self.i_final_embeddings = i_final_embeddings
        
        # user, positive/negative item embedding
        u_emb = u_final_embeddings[u] # shape: (batch size, emb_dim)
        p_emb = i_final_embeddings[p]
        n_emb = i_final_embeddings[n]

        # paper equ. (10)
        y_up = torch.mul(u_emb, p_emb).sum(dim=1) # shape: (batch_size)
        y_un = torch.mul(u_emb, n_emb).sum(dim=1)

        # paper equ. (11): bpr loss + regularization
        bpr_loss = -(torch.log(torch.sigmoid(y_up - y_un))).mean()

        if self.reg > 0: # what's the difference between giving weight decay to Adam?
            l2_norm = (torch.sum(u_emb**2)/2. + torch.sum(p_emb**2)/2. + torch.sum(n_emb**2)/2.) / u_emb.shape[0]
            l2reg = self.reg * l2_norm
            bpr_loss += l2reg
        
        return bpr_loss