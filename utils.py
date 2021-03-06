import os

from torchvision import transforms

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print (pred)
    # print (target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print('correct.shape is {}'.format(correct.shape))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res.append((correct_k, batch_size))
    return res

def train_transform(config):
    return transforms.Compose([
        transforms.Resize([config.img_size, config.img_size]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-135, 135)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def single_transform():
    return transforms.Compose([
        transforms.Resize([64, 64]),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(degrees=(-135, 135)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def val_transform(config):
    return transforms.Compose([
        transforms.Resize([config.img_size, config.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[.5]),
    ])


def get_imgs(root_dir):
    datas = list()
    for roots, dirs, files in os.walk(root_dir):
        for dir in dirs:
            images_name = os.listdir(os.path.join(roots, dir))
            for i in range(len(images_name)):
                image_name = images_name[i]
                datas.append(os.path.join(roots, dir, image_name))
    return datas