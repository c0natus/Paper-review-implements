import torch
import torch.nn as nn
from ddpm import UNet, Diffusion
from utils import dotdict, make_dataloader, train

dict_args = {}
args = dotdict(dict_args)

args.epochs = 500
args.img_size = 64
args.batch_size = 12
args.is_attn = (False, False, True, True)
args.lr = 1e-5
args.save_model_name = 'final'
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.print_pbar = False

dataloader = make_dataloader(args.img_size, args.batch_size)
model = UNet(is_attn=args.is_attn).to(args.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
diffusion = Diffusion(img_size=args.img_size, device=args.device)
train(args, model, diffusion, optimizer, criterion, dataloader)