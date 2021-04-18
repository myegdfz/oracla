
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import get_config
from data_loader import OneData
from utils import train_transform


class dcnn(nn.Module):
    def __init__(self, keep_prob=0.5 ,num_classes=241)->None:
        super(dcnn, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc1 = nn.Sequential(nn.Linear(512*4*4, 1024),  nn.ReLU(), nn.Dropout(.5))
        self.fc2 = nn.Sequential(nn.Linear(1024, num_classes))
    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
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

        self.model = dcnn()
        self.smloss = torch.nn.CrossEntropyLoss()

    def train(self):
        device = self.device
        model = self.model.to(device)
        ceLoss = self.smloss.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_data = OneData(config=self.config, data_dir=self.train_dir, transform=train_transform(self.config))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)
        # print()
        with open('{}/gt.txt'.format(self.model_dir), 'w') as f:
            for step in trange(self.epoch):
                model.train()
                correct = 0
                for idx, (inputs, labels) in enumerate(train_loader):
                    # inputs = torch.cat(inputs, 0)
                    # labels = torch.cat(labels, 0)
                    output, fc1 = model(inputs)
                    # print(output)
                    loss = ceLoss(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    predicted = torch.max(output.data, 1)[1]
                    correct += (predicted == labels).sum()
                    # print(correct)
                    print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}\t Accuracy:{:.6f}'.format(step, idx * len(inputs), len(train_loader.dataset),
                                                                                                 100. * idx / len(train_loader), loss.data.item(), correct / len(train_loader.dataset)))


                print('epoch :{}/{}, loss :{:.6f}'.format(step, self.epoch, loss))

                # with torch.no_grad():
                #     model.eval()
                #     _prec1, _prec3, _batch1, _batch3 = 0, 0, 0, 0
                #     for input, target in valid_loader:
                #         input = input.to(device)
                #         target = target.to(device)
                #         output = model(input)
                #         # output = model.classifier(output)
                #
                #         (prec1, batch1), (prec3, batch3) = accuracy(output.data, target.data, topk=(1, 3))
                #         _prec1 += prec1
                #         _prec3 += prec3
                #         _batch1 += batch1
                #         _batch3 += batch3
                #     print(' * Prec@1 {prec1}/{batch1} Prec@3 {prec3}/{batch3}'.format(prec1=_prec1, batch1=_batch1, prec3=_prec3, batch3=_batch3))

if __name__ == '__main__':
    net = dcnn()
    x = torch.rand((2, 1, 64, 64))

    # print(net)
    # print(net(x).shape)
    print(net(x)[0].shape, net(x)[1].shape)
    config, unparsed = get_config()
    train = Trainer(config)
    train.train()