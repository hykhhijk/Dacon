import os
import numpy as np
import random
import pandas as pd
import argparse
from tqdm.auto import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# dataset
from pretrain_dataset import PretrainDataset

from torch.utils.tensorboard import SummaryWriter

path = "/mnt/d/data/accident/"
sample_submission = pd.read_csv(path+"sample_submission.csv")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataset(val_size=0.1, is_split=False, batch_size=32):
    dataset = PretrainDataset(is_split)
    train_dataset, valid_dataset = dataset.split_dataset(val_size)
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    #valid를 어떤식으로 쪼갤지 생각해보자
    return [train_loader, valid_loader]


def rmsle(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    pred = torch.clamp(pred, min=1e-9)      #간간히 모델이 nan을 반환하는 에러가 나는 이유라 판단하여 추가
    target = torch.clamp(target, min=1e-9)
    loss = torch.square(torch.log1p(pred) - torch.log1p(target))
    return(torch.sqrt(loss.mean() + 1e-9))

def validation(epoch, model, data_loader, criterion, is_split=False):
    print(f'Start validation #{epoch:2d}')
    model.eval()
    losses = 0
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)         
            x = x.to(device)
            y = y.to(device)
            outputs =  model(x)
            if is_split==True:
                y_hat = outputs[:,0]*10 + outputs[:,1]*5 + outputs[:,2]*3 + outputs[:,3]
                loss = criterion(y_hat, y[:,0]*10 + y[:,1]*5 + y[:,2]*3 + y[:,3])
            else:
                loss = criterion(outputs, y)
            # loss = criterion(y, y_hat)

            losses += loss.item()
        # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
        loss = losses / len(data_loader)
        return loss


def train(model, data_loader, criterion, optimizer, epochs, val_every, is_split=False):
    rmsles = 0
    for epoch in range(epochs):
        if epoch < 100:
            criterion = nn.MSELoss()
        else:
            criterion = rmsle
        # loss 초기화
        running_loss = 0
        model.train()
        rmsles=0
        for x, y in data_loader:
            # x, y 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)                
            x = x.to(device)
            y = y.to(device)
            # print(x.shape)
            # print(y.shape)

            optimizer.zero_grad()
            outputs =  model(x)
            if is_split==True:
                y_hat = outputs[:,0]*10 + outputs[:,1]*5 + outputs[:,2]*3 + outputs[:,3]
                loss = criterion(y_hat, y[:,0]*10 + y[:,1]*5 + y[:,2]*3 + y[:,3])
            else:
                loss = criterion(outputs, y)
            loss.backward()

            rmsles += rmsle(outputs, y).item()

            optimizer.step()
            running_loss += loss.item()
        # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
        loss = running_loss / len(data_loader)
        metric = rmsles / len(data_loader)
        # 20번의 Epcoh당 출력합니다.
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/train/rmsle", metric, epoch)
        if epoch % val_every == 0:
            val_loss = validation(epoch=epoch+1, data_loader=valid_loader, criterion=rmsle, model=model, is_split=is_split)
            print("val_loss = {0:.5f}".format(val_loss))
            print("loss = {0:.5f}".format(loss))
            print("train_rmsle = {0:.5f}".format(metric))

            writer.add_scalar("Loss/valid", val_loss, epoch)


    print("----" * 15)
    print("loss = {0:.5f}".format(loss))
    return model


print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))

class DenseModel(nn.Module):
    def __init__(self, input_dim = 10, output_dim=1):
        super(DenseModel, self).__init__()
        self.input_dim = input_dim

        self.layers = []
        self.layers.append(nn.BatchNorm1d(input_dim))
        self.layers.append(nn.Dropout1d())
        self.layers.append(nn.Linear(input_dim, 16, bias=True))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Linear(16, 32, bias=True))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Linear(32, output_dim, bias=True))

        self.net = nn.Sequential(*self.layers)


    def forward(self,x):
        return self.net(x)     
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--is_split", type=str)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    print(args)
    epochs = args.epochs
    val_every = args.val_every
    is_split=False
    batch_size = args.batch_size

    if args.is_split=="True":
        is_split = True

    if is_split==True:
        output_dim = 4
    else:
        output_dim=1
    seed_everything(42)
    train_loader, valid_loader = make_dataset(is_split=is_split, batch_size=batch_size)

    model = DenseModel(13, output_dim).to(device)
    optm = optim.Adam(model.parameters(),lr=1e-5)
    writer = SummaryWriter()

    model = train(model = model, data_loader=train_loader, criterion=rmsle, optimizer=optm, epochs=epochs, val_every=val_every, is_split = is_split)
    torch.save(model.state_dict(), "model_weights.pth")
    writer.flush()


