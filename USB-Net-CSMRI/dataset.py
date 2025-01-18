# %%
import os
import torch
import torchvision
from scipy import io as scio
from torch.utils import data
import glob
import matplotlib.pyplot as plt


class fastmriDataset(data.Dataset):
    def __init__(self, file_path, transforms=None):
        self.images = glob.glob(os.path.join(file_path, '*.mat'))
        self.type = "mat"
        if len(self.images) == 0:
            self.images = glob.glob(os.path.join(file_path, '*.png'))
            self.type = "png"
        self.transforms = transforms

    def __getitem__(self, index):
        if self.type == "mat":
            img = scio.loadmat(self.images[index])['data']
        else:
            img = plt.imread(self.images[index])
        img = torch.from_numpy(img).float()
        img = torch.unsqueeze(img, dim=0)
        if self.transforms is not None:
            img = self.transforms(img)
        label = img
        return img, label

    def __len__(self):
        return len(self.images)


def build_fastmridataset(is_train, args):
    if is_train:
        augs = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(args.input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])
        dataset = fastmriDataset(args.data_path, transforms=augs)
    else:
        augs = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(args.input_size),
        ])
        dataset = fastmriDataset(args.eval_data_path, transforms=augs)
    return dataset
