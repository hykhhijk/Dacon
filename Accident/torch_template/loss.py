import torch
import torch.nn as nn
import torch.nn.functional as F

def rmsle(pred, target):
    pred = pred.contiguous()
    target = target.contiguous()
    pred = torch.clamp(pred, min=1e-9)      #간간히 모델이 nan을 반환하는 에러가 나는 이유라 판단하여 추가
    target = torch.clamp(target, min=1e-9)
    loss = torch.square(torch.log1p(pred) - torch.log1p(target))
    return(torch.sqrt(loss.mean() + 1e-9))

def rmse(pred, target):
    mse = nn.MSELoss()
    rmse = torch.sqrt(mse(pred, target))
    return rmse