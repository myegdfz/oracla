
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torchvision import transforms
from tqdm import tqdm, trange

from config import get_config
from data_loader import OneData, TwoData
from utils import train_transform, val_transform, accuracy

def transforms_tt(img_size):
    return transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-135, 135)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def transforms_tv(img_size):
    return transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        self.valid_dir = config.val_dir
        self.build_model()

    def build_model(self):

        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
        else:
            self.device = torch.device('cpu')

        self.model = vgg16(pretrained=False)
        # print(base_model)
        # dim =  list(base_model.parameters())[-1].shape[0]
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = torch.nn.Sequential(
            nn.Linear(num_ftrs, self.config.num_classes),
            nn.Softmax()
        )

        self.smloss = torch.nn.CrossEntropyLoss()

    def train(self):
        device = self.device
        model = self.model.to(device)
        ceLoss = self.smloss.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_data = TwoData(config=self.config, data_dir=self.train_dir, transform=transforms_tt(224))
        valid_data = TwoData(config=self.config, data_dir=self.valid_dir, transform=transforms_tv(224))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)
        valid_loader = DataLoader(dataset=valid_data, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)
        # print()
        with open('{}/vgg16_gt.txt'.format(self.model_dir), 'w') as f:
            for step in trange(self.epoch):
                model.train()
                correct = 0
                for idx, (inputs, labels) in enumerate(train_loader):
                    # inputs = torch.cat(inputs, 0)
                    # labels = torch.cat(labels, 0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # print(inputs.shape)
                    output = model(inputs)
                    # print(output)
                    loss = ceLoss(output, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    predicted = torch.max(output.data, 1)[1]
                    correct += (predicted == labels).sum()
                    # print(correct)
                    # print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}'.format(step, idx * len(inputs), len(train_loader),
                    #                                                                          100. * idx / len( train_loader), loss.data.item()))
                    print('Epoch :{}[{}/{}({:.0f}%)]\t Loss:{:.6f}\t Accuracy:{:.6f}'.format(step, idx * len(inputs), len(train_loader.dataset),
                                                                                                 100. * (idx / len(train_loader.dataset)), loss.data.item(), correct*1.0 / len(train_loader.dataset)))


                # print('epoch :{}/{}, loss :{:.6f}'.format(step, self.epoch, loss))

                with torch.no_grad():
                    model.eval()
                    _prec1, _prec3, _batch1, _batch3 = 0, 0, 0, 0
                    for input, target in valid_loader:
                        input = input.to(device)
                        target = target.to(device)
                        output = model(input)
                        # output = model.classifier(output)

                        (prec1, batch1), (prec3, batch3) = accuracy(output.data, target.data, topk=(1, 3))
                        _prec1 += prec1
                        _prec3 += prec3
                        _batch1 += batch1
                        _batch3 += batch3
                    print(' * Prec@1 {prec1}/{batch1} Prec@3 {prec3}/{batch3}'.format(prec1=_prec1, batch1=_batch1, prec3=_prec3, batch3=_batch3))


if __name__ == '__main__':

    config, unparsed = get_config()
    train = Trainer(config)
    train.train()