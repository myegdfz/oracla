import json
import os
import random
import shutil

from tqdm import  tqdm
from config import get_config

def get_all_files_path(config):
    root_dir = config.data_dir
    res = []
    dir_list = os.listdir(root_dir)
    for dir in dir_list:
        res.append(os.fspath(root_dir + '/' + dir_list + '/' + dir))
    return res


def get_class(config):
    random.seed(config.random_seed)
    roots_dir = config.data_dir
    classes = set()
    for roots, dirs, files in os.walk(roots_dir):
        for _ in files:
            classes.add(roots)
    print(len(classes))
    classes = list(classes)
    print(classes)
    random.shuffle(classes)

    label_to_subdir = dict()
    label_to_subdir_path = os.path.join(config.data_path, 'label_to_subdir.json')
    subdir_to_label = dict()
    subdir_to_label_path = os.path.join(config.data_path, 'subdir_to_label.json')
    tb = dict()
    for idx, cls in tqdm(enumerate(classes)):
        subdir_to_label[cls[-6:]] = idx
        label_to_subdir[idx] = cls[-6:]

        # target_path = ''
        if idx < int(len(classes) * 0.85):
            target_dir = os.path.join(config.train_dir, cls[-6:])
            # print(target_dir)
            # return 0
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
        else:
            target_dir = os.path.join(config.test_dir, cls[-6:])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        for file in os.listdir(cls):
            file_path = os.path.join(cls, file)
            target_path = os.path.join(target_dir, file)
            shutil.copyfile(file_path, target_path)

        with open(label_to_subdir_path, 'w') as f:
            json.dump(label_to_subdir, f)
        with open(subdir_to_label_path, 'w') as f:
            json.dump(subdir_to_label, f)

def get_class_3(config):
    random.seed(config.random_seed)
    roots_dir = config.data_dir
    classes = set()
    for roots, dirs, files in os.walk(roots_dir):
        for _ in files:
            classes.add(roots)
    print(len(classes))
    classes = list(classes)
    print(classes)
    random.shuffle(classes)
    label_to_subdir = dict()
    label_to_subdir_path = os.path.join(config.data_path, 'label_to_subdir.json')
    subdir_to_label = dict()
    subdir_to_label_path = os.path.join(config.data_path, 'subdir_to_label.json')

    for idx, cls in tqdm(enumerate(classes)):
        subdir_to_label[cls[-6:]] = idx
        label_to_subdir[idx] = cls[-6:]

        # target_path = ''
        if idx < int(len(classes) * 0.8):
            target_dir = os.path.join(config.train_dir, cls[-6:])
            # print(target_dir)
            # return 0
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
        elif  int(len(classes)*0.8) <= idx < int(len(classes) * 0.9):
            target_dir = os.path.join(config.test_dir, cls[-6:])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
        else:
            target_dir = os.path.join(config.val_dir, cls[-6:])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

        for file in os.listdir(cls):
            file_path = os.path.join(cls, file)
            target_path = os.path.join(target_dir, file)
            shutil.copyfile(file_path, target_path)

        with open(label_to_subdir_path, 'w') as f:
            json.dump(label_to_subdir, f)
        with open(subdir_to_label_path, 'w') as f:
            json.dump(subdir_to_label, f)


if __name__ == '__main__':
    config, unparsed = get_config()
    # get_class(config)
    get_class_3(config)