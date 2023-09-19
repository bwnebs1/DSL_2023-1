import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from dataset import *
from util import *

from torchvision import transforms, datasets

import matplotlib.pyplot as plt

## parser
parser = argparse.ArgumentParser(description='Train the UNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lr", default=1e-3, type=float, dest='lr')
parser.add_argument("--batch_size", default=4, type=int, dest='batch_size')
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./results", type=str, dest="result_dir")
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()
## hyperparameter

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dr
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## make dir
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## transfrom and data loading
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

#순서대로 일어남
if mode == "train":
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    ##variables
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## making network
net = UNet().to(device)

## loss function
fn_loss = nn.BCEWithLogitsLoss().to(device)

## optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)

## output functions
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x >0.5)

## summary writer variable
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir= os.path.join(log_dir, 'val'))


## training network
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

if mode =="train": #TRAIN
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # loss function
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # loss function
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input))
            output = fn_tonumpy(fn_class(output))

            writer_val.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_val.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

else: #TEST
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # loss function
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST:  BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'result_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST:  BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_arr)))




