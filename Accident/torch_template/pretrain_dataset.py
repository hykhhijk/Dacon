import pandas as pd 
import os
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
import copy

path = "/mnt/d/data/accident/"

#load_data
train_org = pd.read_csv(os.path.join(path, "external_open/countrywide_accident.csv"))
train_df = train_org.copy()


time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})' 

train_df[['연', '월', '일', '시간']] = train_org['사고일시'].str.extract(time_pattern)
train_df[['연', '월', '일', '시간']] = train_df[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다 
train_df = train_df.drop(columns=['사고일시']) # 정보 추출이 완료된 '사고일시' 컬럼은 제거합니다 

location_pattern = r'(\S+) (\S+) (\S+)'

train_df[['도시', '구', '동']] = train_org['시군구'].str.extract(location_pattern)
train_df = train_df.drop(columns=['시군구'])

road_pattern = r'(.+) - (.+)'

train_df[['도로형태1', '도로형태2']] = train_org['도로형태'].str.extract(road_pattern)
train_df = train_df.drop(columns=['도로형태'])

#split x, y
train_x = train_df[['요일', '기상상태', '노면상태', '사고유형', '연', '월', '일', '시간', '도시', '구', '동',   #baseline전처리만 진행한 test_x의 column
       '도로형태1', '도로형태2']].copy()
train_y = train_df['ECLO'].copy()


categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)
for i in categorical_features:
    le = LabelEncoder()
    le=le.fit(train_x[i]) 
    train_x[i]=le.transform(train_x[i])

class PretrainDataset(Dataset):
    def __init__(self, is_split=False):
        super(PretrainDataset, self).__init__()
        self.x = train_x
        if is_split==True:
            self.y = train_df[['사망자수', '중상자수', '경상자수', '부상자수']]
        else:
            self.y = train_y
        # 텐서 변환
        self.x = torch.tensor(self.x.values).float()
        self.y = torch.tensor(self.y.values).float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    
    def split_dataset(self, val_ratio=0.1):
        train_indices, val_indices = train_test_split(
            range(len(self)), test_size=val_ratio
        )
        self_ = copy.deepcopy(self)
        train_set = Subset(self, train_indices)
        val_set = Subset(self_, val_indices)
        return train_set, val_set
    
# class TestDataset(Dataset):
#     def __init__(self):
#         super(TestDataset, self).__init__()
#         self.x = test_x
#         # 텐서 변환
#         self.x = torch.tensor(self.x.values).float()
        
#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, idx):
#         x = self.x[idx]
#         return x
