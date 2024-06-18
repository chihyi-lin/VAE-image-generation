import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from Model import Model

device = torch.device("cpu")

def train(epochs=12,  # might be worth to tweak this ... orginial=10
          net=Model(),
          batch_size=64  # might be worth to tweak this ...
          ):
    fashion_data = FashionDataset()
    dataloader = DataLoader(dataset=fashion_data,
                            batch_size=batch_size,
                            num_workers=2,
                            drop_last=True,
                            prefetch_factor=6)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)  # might be worth to tweak this ...

    for i in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(tqdm(dataloader)):
            # batch has a shape of [b, 1, 28, 28]
            # put the batch into the net
            x = x.view(batch_size, 784)
            x = x.to(device)
            optimizer.zero_grad()

            reconstructions_of_targets, kl_loss, reconstruction_loss = net(x)
            # kl_loss * 0.1 -> as kl_loss decreases very fast
            loss = kl_loss + reconstruction_loss
            overall_loss += loss.item()

            # backpropagate and call the optimizer
            loss.backward()
            optimizer.step()
        print("\tEpoch", i+1, "complete", "\tAverage Loss: ", overall_loss / len(dataloader))
    
    print("Finish")

    torch.save({"model": net.state_dict()}, f="checkpoint.pt")  # saving a checkpoint for later use during sampling.
    # CAREFUL: will overwrite previous checkpoint


class FashionDataset(Dataset):
    def __init__(self):
        fashion_mnist = FashionMNIST(root=".", download=True)
        self.imgs = list()
        for el in fashion_mnist:
            self.imgs.append(el[0])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        return self.transform(self.imgs[index]).float()

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    train()
