import os
import csv
import random
import pathlib
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import onn
from dataset import MyDataset
from pathlib import Path


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main(args):

    split = 0.8
    x_data = list(Path('data/out2/').glob('*.jpg'))
    totalNum = len(x_data)

    train_x_data = x_data[:int(totalNum*split)]
    test_x_data = x_data[int(totalNum*split):]

    train_data = MyDataset(train_x_data,'data/LTrain/')
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=16,shuffle=True)

    SIZE = 256
    padding = int((SIZE-128)/2)
    model = onn.Net(SIZE, num_layers=30)
    model.cuda()


    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


    for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):

        log = [epoch]

        model.train()

        train_running_loss = 0.0

        for train_iter,train_data_batch in enumerate(train_loader):

            train_source = train_data_batch[0].cuda()       
            train_target = train_data_batch[1].cuda()   

            train_source = F.pad(train_source, pad=(padding, padding, padding, padding))

            train_source = torch.squeeze(torch.cat((train_source.unsqueeze(-1),torch.zeros_like(train_source.unsqueeze(-1))), dim=-1), dim=1)

            optimizer.zero_grad()
            train_outputs = model(train_source)
            train_loss_ = criterion(train_outputs, train_target)
            train_loss_.backward()
            optimizer.step()

            train_running_loss += train_loss_.item()
            mse = criterion(train_outputs.detach(),train_target.detach())

            #train_loss = train_running_loss / (train_iter+1)
            print('epoch %d batch %d loss %.6f lr_rate %.10f'%(epoch, train_iter, train_loss_,optimizer.param_groups[0]['lr']))
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--whether-load-model', type=bool, default=False, help="是否加载模型继续训练")
    parser.add_argument('--start-epoch', type=int, default=0, help='从哪个epoch继续训练')
    # 数据和模型相关
    parser.add_argument('--model-name', type=str, default='_model.pth')
    parser.add_argument('--model-save-path', type=str, default="./saved_model/")
    parser.add_argument('--result-record-path', type=pathlib.Path, default="./result.csv", help="数值结果记录路径")

    torch.backends.cudnn.benchmark = True
    args_ = parser.parse_args()
    random.seed(args_.seed)
    np.random.seed(args_.seed)
    torch.manual_seed(args_.seed)
    main(args_)
