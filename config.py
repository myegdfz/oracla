import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data Process
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, default='./data')
data_arg.add_argument('--data_path', type=str, default='./net')
data_arg.add_argument('--train_dir', type=str, default='./net/train')
data_arg.add_argument('--test_dir', type=str, default='./net/test')
data_arg.add_argument('--val_dir', type=str, default='/d6295745ef534beab3ce2490bedcd8ab/hy/netvlad/data/val')
data_arg.add_argument('--random_seed', type=int, default=20)
data_arg.add_argument('--num_classes', type=int, default=181)
data_arg.add_argument('--num_size', type=int, default=50)
data_arg.add_argument('--img_size', type=int, default=64)
# Model Parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--base_model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'alexnet', 'vgg16', 'googlenet'])
model_arg.add_argument('--model_path', type=str, default='/d6295745ef534beab3ce2490bedcd8ab/hy/netvlad')
model_arg.add_argument('--model_dir', type=str, default='./model')
model_arg.add_argument('--num_clusters', type=int, default=32)
model_arg.add_argument('--vlad_alpha', type=float, default=1.0)
# Train Process
train_arg = add_argument_group('Train')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--use_vlad', type=str2bool, default=True)
train_arg.add_argument('--gpu', type=int, default=0)
train_arg.add_argument('--batch_size', type=int, default=64)
train_arg.add_argument('--num_workers', type=int, default=0)
train_arg.add_argument('--lr', type=float, default=0.005)
train_arg.add_argument('--lr_decay', type=float, default=0.5)
train_arg.add_argument('--lr_update_step', type=int, default=10)
train_arg.add_argument('--lr_lower_boundary', type=float, default=0.00001)
train_arg.add_argument('--momentum', type=float, default=0.9)
train_arg.add_argument('--weight_decay', type=float, default=1e-4)
train_arg.add_argument('--epoch', type=int, default=100)
train_arg.add_argument('--tmloss_margin', type=float, default=1.0)
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
train_arg.add_argument('--log_dir', type=str, default='/d6295745ef534beab3ce2490bedcd8ab/hy/netvlad/log')
train_arg.add_argument('--tag', type=str, default='test')
# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--load_model', type=str, default='')


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed
