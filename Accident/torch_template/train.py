import os
import numpy as np
import random
import pandas as pd
from tqdm.auto import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
# dataset
from dataset import CustomDataset
from dataset import TestDataset

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


def make_dataset(val_size=0.1, is_split=False):
    dataset = CustomDataset(is_split)
    train_dataset, valid_dataset = dataset.split_dataset(val_size)
    test_dataset = TestDataset()
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


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
    train_losses = []
    for epoch in range(epochs):
        # loss 초기화
        running_loss = 0
        model.train()
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
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 누적합산된 배치별 loss값을 배치의 개수로 나누어 Epoch당 loss를 산출합니다.
        loss = running_loss / len(data_loader)
        train_losses.append(loss)

        # 20번의 Epcoh당 출력합니다.
        if epoch % val_every == 0:
            val_loss = validation(epoch=epoch+1, data_loader=valid_loader, criterion=criterion, model=model, is_split=is_split)
            print("{0:05d} val_loss = {1:.5f}".format(epoch, val_loss))
            print("{0:05d} loss = {1:.5f}".format(epoch, loss))


    print("----" * 15)
    print("{0:05d} loss = {1:.5f}".format(epoch, loss))


print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))

class DenseModel(nn.Module):
    def __init__(self, input_dim = 10, output_dim=1):
        super(DenseModel, self).__init__()
        self.input_dim = input_dim

        self.layers = []
        self.layers.append(nn.BatchNorm1d(input_dim))
        self.layers.append(nn.Linear(input_dim, 16, bias=True))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Linear(16, 32, bias=True))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.Linear(32, output_dim, bias=True))

        self.net = nn.Sequential(*self.layers)


    def forward(self,x):
        return self.net(x)     
    
val_every = 5
is_split=True
train_loader, valid_loader, test_loader = make_dataset(is_split=is_split)

model = DenseModel(20, 4).to(device)
optm = optim.Adam(model.parameters(),lr=1e-4)

train(model = model, data_loader=train_loader, criterion=rmsle, optimizer=optm, epochs=100, val_every=val_every, is_split = is_split)

predictions = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch_in = batch.to(device)
        pred_ = model(batch_in)
        pred = pred_[:,0]*10 + pred_[:,1]*5 +pred_[:,2]*3 + pred_[:,3]
        predictions.append(pred.cpu().numpy())
all_predictions = np.concatenate(predictions, axis=0)

baseline_submission = sample_submission.copy()
baseline_submission['ECLO'] = all_predictions

baseline_submission.to_csv('baseline_submit.csv', index=False)
