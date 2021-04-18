import json
import os

import torch
from PIL import Image
from config import get_config
from torch.utils.data import Dataset
from utils import train_transform
import numpy as np
torch.set_printoptions(threshold=np.inf)

class OneData(Dataset):
    def __init__(self, config, data_dir, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.load_dir(config.data_path)
        self.get_img(self.data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        image_path, label = self.data_map[idx]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def load_dir(self, data_path):
        dir2label = os.path.join(data_path, 'subdir_to_label.json')
        with open(dir2label, 'r') as f:
            self.dir2label = json.load(f)

    def get_img(self, data_dir):
        self.data_map = list()
        for roots, dirs, files in os.walk(data_dir):
            for dir in dirs:
                image_names = os.listdir(os.fspath(roots + "/" + dir))
                for i in range(len(image_names)):
                    image_name = image_names[i]
                    image_path = os.path.join(roots, dir, image_name)
                    self.data_map.append((image_path, self.dir2label[dir]))

class TwoData(Dataset):

    def __init__(self, config, data_dir, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.load_dir(config.data_path)
        self.get_img(self.data_dir)
        self.transform = transform

    def load_dir(self, data_path):
        pass


if __name__ == '__main__':
    config, unparsed = get_config()
    dataset = OneData(config, data_dir=config.train_dir, transform=train_transform(config))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.num_workers)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # total_inputs = torch.cat(inputs, 0)
        # print(total_inputs)
        print(labels)
        print(inputs[[0][0]])
        break

