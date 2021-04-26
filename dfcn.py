import os

import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import get_config
from data_loader import OneData
from extract_features import DFCN_DCNN
from utils import train_transform, val_transform, accuracy



class dfcn(nn.Module):
    def __init__(self, keep_prob=0.5, num_classes=241):
        super(dfcn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc11 = nn.Sequential(nn.Linear(512 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5))
        self.fc1 = nn.Sequential( nn.Linear(1024, 2048), nn.ReLU(), )
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
        x = self.features(x)
        x = self.fc11(x)
        fc11 = x
        fc1 = self.fc1(x)
        x = self.fc2(fc1)
        return x, fc1, fc11

class EmbedNet(nn.Module):
    def __init__(self, base_model, dfcn):
        super(EmbedNet, self).__init__()
        self.base_model = base_model
        self.dfcn = dfcn

    def forward(self, x):
        _, fc1 = self.base_model(x)
        output, dfcn_fc1 = self.dfcn(fc1)
        return output, dfcn_fc1

class Trainer(object):
    def __init__(self, config)-> None:
        super().__init__()
        self.config = config
        self.base_model = config.base_model
        self.model_dir = config.model_dir
        self.lr = config.lr
        self.epoch = 1
        self.batch_size = config.batch_size
        self.gpu_id = config.gpu

        self.num_classes = config.num_classes
        self.train_dir = config.train_dir
        self.valid_dir = config.val_dir
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

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_data = OneData(config=self.config, data_dir=self.train_dir, transform=train_transform(self.config))
        valid_data = OneData(config=self.config, data_dir=self.valid_dir, transform=val_transform(self.config))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.config.num_workers)
        valid_loader = DataLoader(dataset=valid_data, batch_size=self.batch_size, num_workers=self.config.num_workers)
        # print()
        with open('{}/dfcn_gt.txt'.format(self.model_dir), 'w') as f:
            for step in trange(self.epoch):
                model.train()
                # correct = 0
                for idx, (inputs, labels) in enumerate(train_loader):
                    # inputs = torch.cat(inputs, 0)
                    # labels = torch.cat(labels, 0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    output, fc1, fc11 = model(inputs)
                    # print(output)
                    loss = ceLoss(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    predicted = torch.max(output.data, 1)[1]
                    # correct += (predicted == labels).sum()
                    # print(correct)
                    # print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}'.format(step, idx * len(inputs), len(train_loader),
                    #                                                                          100. * idx / len(train_loader), loss.data.item()))


                with torch.no_grad():
                    model.eval()
                    _prec1, _prec3, _batch1, _batch3 = 0, 0, 0, 0
                    correct = 0
                    for input, target in valid_loader:
                        input = input.to(device)
                        target = target.to(device)
                        # print('input', input)
                        # print('target', target)
                        output, fc1, fc11 = model(input)
                        # predicted = torch.max(output.data, 1)[1]
                        # correct += (target == predicted).sum()
                        # output = model.classifier(output)
                        # print('output', output)
                        # return 0
                        (prec1, batch1), (prec3, batch3) = accuracy(output.data, target.data, topk=(1, 3))
                        _prec1 += prec1
                        _prec3 += prec3
                        _batch1 += batch1
                        _batch3 += batch3
                    print(' * Prec@1 {prec1}/{batch1} Prec@3 {prec3}/{batch3}'.format(prec1=_prec1, batch1=_batch1, prec3=_prec3, batch3=_batch3))
        torch.save(model, self.config.model_path)

def is_accuracy_k(name, scores_name):
    # print(name)
    # print(scores_name)
    name = os.path.split(name)[-1]
    for it in scores_name[1:]:
        if name[:6] == it[:6].decode('utf-8'):
            return 1
    return 0

def get_imgs(root_dir):
    datas = list()
    for roots, dirs, files in os.walk(root_dir):
        for dir in dirs:
            images_name = os.listdir(os.path.join(roots, dir))
            for i in range(len(images_name)):
                image_name = images_name[i]
                datas.append(os.path.join(roots, dir, image_name))
    return datas

def read_h5(index_path):
    h5f = h5py.File(index_path, 'r')
    # feats = h5f['dataset_1'][:]
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    # print(imgNames)
    h5f.close()
    return feats, imgNames

def work(test, query,mode, max_size, feats, imgNames):
    query = test.extract_feat(query, mode)
    scores = np.dot(query, feats.T)
    rank_ID = np.argsort(scores)[::-1]
    maxres = max_size
    imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
    return imlist


selected = 1000
img_list = get_imgs('net/test')
img_list = np.array(img_list)
qlist = np.random.choice(img_list, selected)

idx_list = [('dfcn', 'featsDFCN.h5')]

if __name__ == '__main__':
    # net = dfcn()
    #
    # x = torch.rand((2, 1, 64, 64))
    # # print(net)
    # # print(net(x).shape)
    # print(net(x)[0].shape, net(x)[1].shape)
    config, unparsed = get_config()
    # train = Trainer(config)
    # train.train()
    # extract image features
    model = torch.load(config.model_path, map_location='cpu')
    test = DFCN_DCNN(model, config)
    # test.eval_model()
    # test.extract_feat('net/test/001027/001027b00024 pos.bmp', 'dfcn')
    for idx, db in tqdm(idx_list):
        feats, img_names = read_h5(db)
        rate = 0
        for query in qlist:
            # print('query image is', query, 'method is', idx)
            imlist = work(test, query, idx, 3, feats, img_names)
            # print('test', imlist)
            if is_accuracy_k(query, imlist):
                rate += 1
            # print(query, imlist)
        print("{}, accuracy:{:.6f}".format(idx, rate / selected))
