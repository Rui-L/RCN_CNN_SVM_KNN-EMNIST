#!/usr/bin/env python3
# coding: utf-8

""" CNN on EMNIST-letters dataset. """
__author__ = "Rui Lin"
__email__ = "rxxlin@umich.edu"


################ initialization ###############
import os

# -- torch
import torch
from torch import autograd
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

torch.manual_seed(42)
Var = autograd.Variable
T = torch.Tensor

# -- PIL
from PIL import Image

# -- logging
import logging
import logging.handlers
LOG_LEVEL = logging.INFO

def logging_config():
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    f = logging.Formatter("[%(levelname)s]%(module)s->%(funcName)s: \t %(message)s \t --- %(asctime)s")

    h = logging.StreamHandler()
    h.setFormatter(f)
    h.setLevel(LOG_LEVEL)

    root_logger.addHandler(h)


# -- argparse
import argparse

def parse_cmd():
    parser = argparse.ArgumentParser(description='CNN for EMNIST')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_per_class_training', type=int, default=None)
    parser.add_argument('--num_per_class_testing', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    return args


# -- dataset
class EMNIST(Dataset):
    def __init__(self, root_dir, num_per_class=None, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])):
        self.root_dir = root_dir
        self.num_per_class = num_per_class
        self.transform = transform
        self.paths = []
        self.cls_list = sorted(os.listdir(root_dir))
        self.num_cls = len(self.cls_list)
        for cls in self.cls_list:
            for img in sorted(os.listdir(os.path.join(self.root_dir, cls)))[:self.num_per_class]:
                self.paths.append((cls, img))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, *self.paths[idx])
        image = Image.open(img_name).convert(mode="L")
        label = self.cls_list.index(self.paths[idx][0])

        if self.transform:
            image = self.transform(image)

        return image, label


# -- CNN, use the official pytorch example as reference,
# to make sure we can get a reasonable result to compare with RCN
# (https://github.com/pytorch/examples/blob/master/mnist/main.py)
class Net(nn.Module):
    def __init__(self, num_cls):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d() # avoid overfitting
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_cls)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# -- training
def train1epoch(model, loader):
    model.train()
    total_loss = 0
    batch_cnt = 0
    for batch_idx, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Var(data), Var(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]
        batch_cnt += 1

        # release GPU memory
        del loss
        del output
        del data
        del target

    return total_loss / batch_cnt


# -- testing
def test(model, loader):
    model.eval() ## set Dropout / BatchNorm in evaluation mode
    test_loss = 0
    batch_cnt = 0
    correct = 0
    for data, target in loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Var(data, volatile=True), Var(target)
        output = model(data)

        test_loss += F.nll_loss(output, target).cpu().data[0] # sum up average batch loss
        batch_cnt += 1

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # release GPU memory
        del pred
        del output
        del data
        del target

    return test_loss / batch_cnt, correct / len(loader.dataset)


if __name__ == "__main__":
    logging_config()
    logger = logging.getLogger()

    args = parse_cmd()

    # data set
    dir_training = os.path.join(args.data_dir, 'training')
    dir_testing = os.path.join(args.data_dir, 'testing')
    train_loader = DataLoader(EMNIST(dir_training, args.num_per_class_training), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(EMNIST(dir_testing, args.num_per_class_training), batch_size=args.batch_size, shuffle=True, num_workers=4)

    # model
    model = Net(train_loader.dataset.num_cls)
    if torch.cuda.is_available():
        model.cuda()

    # training
    optimizer = optim.Adam(model.parameters()) # using default lr=0.001, betas=(0.9, 0.999)
    for e in range(args.epochs):
        loss = train1epoch(model, train_loader)
        logger.info("Epoch %s: %s", e, loss)
        if e % 5 == 0:
            loss, acc = test(model, test_loader)
            logger.info("test loss: %s, test accuracy: %s", loss, acc)
    loss, acc = test(model, test_loader)
    logger.info("test loss: %s, test accuracy: %s", loss, acc)

