import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import activation_getter

class Caser(nn.Module):
    def __init__(self, num_users, num_items, model_args):
        """
        Args:
            num_users (int): user의 수
            num_items (int): item의 수
            model_args (args): 모델과 관련된 argument like latent dimension
        """
        super(Caser, self).__init__()
        self.args = model_args
        L = self.args.L
        dims = self.args.d # latent dimension
        self.n_h = self.args.nh # number of filter in horizontal convolution
        self.n_v = self.args.nv # number of filter in vertical convolution
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # horizontal conv layers
        heights = [i+1 for i in range(L)]
        self.conv_h = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(
                    in_channels=1, 
                    out_channels=self.n_h, # number of filters
                    kernel_size=(i, dims) # filter size
                    ),
                self.ac_conv()
                ) for i in heights]
            )

        # vertical conv layers
        self.conv_v = nn.Conv2d(
            in_channels=1, 
            out_channels=self.n_v, 
            kernel_size=(L, 1)
            )

        # fully-connected layer
        self.fc1_dim_h = self.n_h * len(heights) # filter 개수 x 사용한 hegiht: conv 결과를 maxpooling 하므로
        self.fc1_dim_v = self.n_v * dims # filter 개수 x latent dims: conv 연산 결과 latent dims의 vector이므로
        fc1_dim_in = self.fc1_dim_h + self.fc1_dim_v
        
        self.fc1_layer = nn.Sequential(
            nn.Dropout(self.drop_ratio),
            nn.Linear(fc1_dim_in, dims), # default: bias=True
            self.ac_fc()
            )

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scopredict for all items
        # 만약 nn.Linear로 계산하면 모든 item에 대한 확률을 계산한다.
        # 하지만, positive sampling과 negative sampling만을 예측하는 것이 더 효율적이다.
        self.W2 = nn.Embedding(num_items, dims+dims)
        self.b2 = nn.Embedding(num_items, 1)

        self._init_weights()
    

    def _init_weights(self):
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    
    def forward(self, user_var, seq_var, item_var, for_pred=False):
        """
        The forward propagation used to get recommendation scopredict, given
        triplet (user, sequence, targets).
        
        Args:
            seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
                a batch of sequence
            user_var: torch.LongTensor with size [batch_size]
                a batch of user
            item_var: torch.LongTensor with size [batch_size, L]
                a batch of items
            for_pred: boolean, optional
                Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_var)

        # Convolutional Layers
        # weight (4-D): [out_channels, in_channels, L, 1]
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = conv(item_embs).squeeze(3)
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers's input
        out = torch.cat([out_v, out_h], 1)

        # fully-connected layer
        z = self.fc1_layer(out)
        x = torch.cat([z, user_emb], 1)

        # item_var = pos_items(T) + num_neg_samples
        # Embedding Look up
        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            # top K를 뽑을 것이기 때문에 sigmoid 필요 없다.
            predict = (x * w2).sum(1) + b2
        else:
            # torch.baddbmm:
            #   batch matrix-matrix product of matrices w2, x.
            #   b2 is added to the final result
            predict = torch.sigmoid(torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze())

        return predict