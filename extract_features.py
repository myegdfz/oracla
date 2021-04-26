import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import h5py

from config import get_config
from PIL import Image
import numpy as np

from data_loader import OneData
from utils import train_transform, val_transform, accuracy, get_imgs, single_transform
np.set_printoptions(1000000)

class DFCN_DCNN:
    def __init__(self,model, config) -> None:
        super(DFCN_DCNN, self).__init__()
        self.model = model
        self.config = config
        self.gpu_id = config.gpu
        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
        else:
            self.device = torch.device('cpu')
        self.train_dir = config.test_dir
        self.batch_size = config.batch_size
        self.path = "path"

    def eval_model(self):
        device = self.device
        model = self.model.to(device)
        # model.load_state_dict(torch.load(self.path))
        # model.eval()

        dfcn_feats, dcnn_feats = [], []
        names = list()
        h5f1 = h5py.File('featsDCNN.h5', 'w')
        h5f2 = h5py.File('featsDFCN.h5', 'w')
        imgs = get_imgs('net/train')

        train_data = OneData(config=self.config, data_dir=self.train_dir, transform=train_transform(self.config))
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.config.num_workers)
        # print(imgs)
        for idx, (inputs, labels) in enumerate(train_loader):
            # name = os.path.split(inputs)[-1]
            # inputs = inputs[np.newaxis, :]
            inputs = inputs.to(device)
            # labels = labels.to(device)
            # print(labels)
            # print(inputs[0])

            output, dfcn, dcnn = model(inputs)
            # dfcn = np.array(dfcn)
            # dfcn = np.squeeze(dfcn)
            # dcnn = np.squeeze(dcnn)
            # print(dfcn.shape)
            # print(dcnn.shape)
            for it in dfcn:
                tmp = list()
                for i in it:
                    tmp.append(float(i))
                #
                # it = list(it)
                # it = np.squeeze(it)
                # print(it.shape)
                dfcn_feats.append(tmp)

            for it in dcnn:
                tmp = list()
                for i in it:
                    tmp.append(float(i))
                dcnn_feats.append(tmp)
            # return
            # dfcn_feats.append(dfcn)
            # dcnn_feats.append(dcnn)
            print("extracting feature from image No. %d , %d images in total" % ((idx + 1), len(train_loader)))
            for name in labels:
                names.append(name)
            # names.append(name)
            # if idx == 9:
            #     break
        # print(dcnn_feats)
        # print(np.array(dcnn_feats).shape)

        feats = np.array(dcnn_feats)
        print(feats)
        print(feats.shape)
        # return
        h5f1.create_dataset('dataset_1', data=feats)
        feats = np.array(dfcn_feats)
        h5f2.create_dataset('dataset_1', data=feats)
        names = [name.encode() for name in names]
        h5f1.create_dataset('dataset_2', data=np.string_(names))
        h5f2.create_dataset('dataset_2', data=np.string_(names))
        h5f1.close()
        h5f2.close()

    def extract_feat(self, image_path, mode):
        img =  Image.open(image_path).convert('L')
        img = img.resize((64, 64))
        to_tensor = single_transform()
        img = to_tensor(img)
        img = img.unsqueeze(0)
        _, dfcn, dcnn = self.model(img)
        with torch.no_grad():
            dfcn = dfcn.numpy()
            dfcn = dfcn.squeeze()
            dcnn = dcnn.numpy().squeeze()
            # print(dcnn.shape, dfcn.shape)
            if mode == 'dcnn':
                return dcnn
            else: return dfcn


if __name__ == '__main__':
    config, unparsed = get_config()
    # model = torch.load(config.model_path, map_location='cpu')
    # mode = DFCN_DCNN(model, config)
    # # mode.eval_model()
    # mode.extract_feat('net/test/001027/001027b00024 pos.bmp')

