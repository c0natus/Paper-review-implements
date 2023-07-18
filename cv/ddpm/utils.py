import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from datetime import timedelta


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_transformed_dataset(img_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)), # height, width
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root="./", download=False,
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root="./", download=False,
                                         transform=data_transform, split='test')
    return train, test


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def make_dataloader(img_size, batch_size):
    
    data_train, data_test = load_transformed_dataset(img_size)
    dataset =  torch.utils.data.ConcatDataset([data_train, data_test])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_one_epoch(args, model, diffusion, dataloader, optimizer, criterion):
    if args.print_pbar: pbar = tqdm(dataloader)
    else: pbar = dataloader
    total_loss = 0
    for images, _ in pbar:
        images = images.to(args.device)
        t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
        x_t, noise = diffusion.noise_images(images, t)
        predicted_noise = model(x_t, t) 
        loss = criterion(noise, predicted_noise)
        # noise를 없앤 image 자체를 학습하고 싶으면 x_t와 predicted_noise를 loss의 input으로 준다.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.print_pbar: pbar.set_postfix(MSE=f'{loss.item():.3f}')
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss
    

def train(args, model, diffusion, optimizer, criterion, dataloader):
    print('Start Training')
    model.train()
    for epoch in range(args.epochs):
        start = time()
        avg_loss = train_one_epoch(args, model, diffusion, dataloader, optimizer, criterion)
        print(f'Epoch: [{epoch:03d}/{args.epochs}] \t loss={avg_loss:.4f} \t Took: {str(timedelta(seconds=int(time() - start)))}')
        torch.save(model, args.save_model_name + '.pt')
    print('Finish!')