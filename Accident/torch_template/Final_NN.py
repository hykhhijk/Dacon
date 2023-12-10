import os
import numpy as np
import random
import pandas as pd
import argparse
from tqdm.auto import tqdm
from importlib import import_module

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# dataset
from dataset import CustomDataset
from dataset import TestDataset

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
    dataset = CustomDataset(is_split)
    train_dataset, valid_dataset = dataset.split_dataset(val_size)
    test_dataset = TestDataset()
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    #valid를 어떤식으로 쪼갤지 생각해보자
    return [train_loader, valid_loader, test_loader]


def rmsle(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    pred = torch.clamp(pred, min=1e-9)      #간간히 모델이 nan을 반환하는 에러가 나는 이유라 판단하여 추가
    target = torch.clamp(target, min=1e-9)
    loss = torch.square(torch.log1p(pred) - torch.log1p(target))
    return(torch.sqrt(loss.mean() + 1e-9))

def validation(epoch, model, data_loader, criterion, is_split=False):
    print(f'Start validation #{epoch:2d}{"----"*10}')
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
    #append load model_weights
    model.load_state_dict(torch.load("model_weights.pth"))


    for epoch in range(epochs):
        # loss 초기화
        running_loss = 0
        model.train()
        rmsles=0
        for x, y in data_loader:
            # x, y 데이터를 device 에 올립니다. (cuda:0 혹은 cpu)                
            x = x.to(device)
            y = y.to(device)
        
            optimizer.zero_grad()
            outputs =  model(x)
            if is_split==True:
                y_hat = outputs[:,0]*10 + outputs[:,1]*5 + outputs[:,2]*3 + outputs[:,3]
                y = y[:,0]*10 + y[:,1]*5 + y[:,2]*3 + y[:,3]
                loss = criterion(y_hat, y)
            else:
                loss = criterion(outputs, y)

            if args.loss!="rmsle":
                rmsles += rmsle(outputs, y).item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
        loss = running_loss / len(data_loader)
        rmsles = rmsles / len(data_loader)
        if write=="True":
            writer.add_scalar("Loss/train", loss, epoch)
        if args.loss!="rmsle":
            print(f"rmsle: {rmsles}")
            if write=="True":
                writer.add_scalar("Loss/train/rmsle", rmsles, epoch)
        # 20번의 Epcoh당 출력합니다.
        if epoch % val_every == 0:
            val_loss = validation(epoch=epoch+1, data_loader=valid_loader, criterion=criterion, model=model, is_split=is_split)
            print("val_loss = {0:.5f}".format(val_loss))
            print("loss = {0:.5f}".format(loss))
            if write=="True":
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("Loss/valid", val_loss, epoch)


    print("----" * 15)
    print("loss = {0:.5f}".format(loss))


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


        self.net = nn.Sequential(*self.layers)


    def forward(self,x):
        return self.net(x)     
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_name", type=str, default="baseline_submit")
    parser.add_argument("--loss", type=str, default="rmsle")
    parser.add_argument("--write", type=str, default="True")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--is_split", type=str)



    args = parser.parse_args()
    print(args)
    epochs = args.epochs
    val_every = args.val_every
    batch_size = args.batch_size
    base_name = args.base_name
    write = args.write
    suffix = args.suffix

    criterion = getattr(import_module("loss"), args.loss)

    is_split=False
    if args.is_split=="True":
        is_split = True

    if is_split==True:
        output_dim = 4
    else:
        output_dim=1
    seed_everything(42)
    train_loader, valid_loader, test_loader = make_dataset(is_split=is_split, batch_size=batch_size)

    model = DenseModel(13, output_dim).to(device)
    optm = optim.Adam(model.parameters(),lr=1e-3)
    if write=="True":
        writer = SummaryWriter(comment=f"batch_size_{batch_size}_{suffix}")         #config this line for experiment value
    train(model = model, data_loader=train_loader, criterion=criterion, optimizer=optm, epochs=epochs, val_every=val_every, is_split = is_split)
    if write=="True":
        writer.flush()

    #pordict test set
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch_in = batch.to(device)
            pred_ = model(batch_in)
            if is_split==True:
                pred = pred_[:,0]*10 + pred_[:,1]*5 +pred_[:,2]*3 + pred_[:,3]
            else:
                pred = pred_
            predictions.append(pred.cpu().numpy())
    all_predictions = np.concatenate(predictions, axis=0)

    baseline_submission = sample_submission.copy()
    baseline_submission['ECLO'] = all_predictions
    save_name = base_name
    cnt=0
    while os.path.isfile(save_name+".csv"):
        save_name = f"{base_name}_{cnt}"
        cnt+=1
    baseline_submission.to_csv(save_name+".csv", index=False)

