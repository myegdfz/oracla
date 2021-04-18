import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import get_config
from data_loader import OneData
from utils import train_transform

class dfcn(nn.Module):
    def __init__(self, keep_prob=0.5, num_classes=241):
        super(dfcn, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(1.0),
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(keep_prob),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(keep_prob),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )
    def forward(self, x):
        fc1 = self.fc1(x)
        x = self.fc2(fc1)
        return x, fc1


class Trainer(object):
    def __init__(self, config)-> None:
        super().__init__()
        self.config = config
        self.model_dir = config.model_dir
        self.lr = config.lr
        self.epoch = 20
        self.batch_size = config.batch_size
        self.gpu_id = config.gpu

        self.num_classes = config.num_classes
        self.train_dir = config.train_dir
        self.build_model()

    def build_model(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
        else:
            self.device = torch.device('cpu')

        self.model = dfcn()
        self.smloss = torch.nn.CrossEntropyLoss()

    def train(self):
        device = self.device
        model = self.model.to(device)
        ceLoss = self.smloss.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_data = OneData(config=self.config, data_dir=self.train_dir, transform=train_transform(self.config))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)
        # print()

if __name__ == '__main__':
    net = dfcn()
    x = torch.rand((2, 1, 64, 64))

    # print(net)
    # print(net(x).shape)
    print(net(x)[0].shape, net(x)[1].shape)
    config, unparsed = get_config()
    # train = Trainer(config)
    # train.train()
