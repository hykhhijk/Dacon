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
train_org = pd.read_csv(path + 'train.csv') 
test_org = pd.read_csv(path + 'test.csv')
train_df = train_org.copy()
test_df = test_org.copy()


#preprocess
time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})' 
train_df[['연', '월', '일', '시간']] = train_org['사고일시'].str.extract(time_pattern)
train_df[['연', '월', '일', '시간']] = train_df[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다 
train_df = train_df.drop(columns=['사고일시']) # 정보 추출이 완료된 '사고일시' 컬럼은 제거합니다 
test_df[['연', '월', '일', '시간']] = test_org['사고일시'].str.extract(time_pattern)
test_df[['연', '월', '일', '시간']] = test_df[['연', '월', '일', '시간']].apply(pd.to_numeric)
test_df = test_df.drop(columns=['사고일시'])

location_pattern = r'(\S+) (\S+) (\S+)'
train_df[['도시', '구', '동']] = train_org['시군구'].str.extract(location_pattern)
train_df = train_df.drop(columns=['시군구'])
test_df[['도시', '구', '동']] = test_org['시군구'].str.extract(location_pattern)
test_df = test_df.drop(columns=['시군구'])

road_pattern = r'(.+) - (.+)'
train_df[['도로형태1', '도로형태2']] = train_org['도로형태'].str.extract(road_pattern)
train_df = train_df.drop(columns=['도로형태'])
test_df[['도로형태1', '도로형태2']] = test_org['도로형태'].str.extract(road_pattern)
test_df = test_df.drop(columns=['도로형태'])

#additional preprocess
light_df = pd.read_csv(os.path.join(path, "external_open/light.csv"), encoding='cp949')[['설치개수', '소재지지번주소']]
location_pattern = r'(\S+) (\S+) (\S+) (\S+)'
light_df[['도시', '구', '동', '번지']] = light_df['소재지지번주소'].str.extract(location_pattern)
light_df = light_df.drop(columns=['소재지지번주소', '번지'])
light_df = light_df.groupby(['도시', '구', '동']).sum().reset_index()
light_df.reset_index(inplace=True, drop=True)

child_area_df = pd.read_csv(os.path.join(path, "external_open/child.csv"), encoding='cp949')[['CCTV설치대수', '소재지지번주소']]
child_area_df['보호구역수'] = 1
location_pattern = r'(\S+) (\S+) (\S+) (\S+)'
child_area_df[['도시', '구', '동', '번지']] = child_area_df['소재지지번주소'].str.extract(location_pattern)
child_area_df = child_area_df.drop(columns=['소재지지번주소', '번지'])
child_area_df = child_area_df.groupby(['도시', '구', '동']).sum().reset_index()
child_area_df.reset_index(inplace=True, drop=True)

parking_df = pd.read_csv(os.path.join(path, "external_open/parking.csv"), encoding='cp949')[['소재지지번주소', '급지구분', "주차구획수"]]
parking_df = pd.get_dummies(parking_df, columns=['급지구분'])
location_pattern = r'(\S+) (\S+) (\S+) (\S+)'
parking_df[['도시', '구', '동', '번지']] = parking_df['소재지지번주소'].str.extract(location_pattern)
parking_df = parking_df.drop(columns=['소재지지번주소', '번지'])
parking_df = parking_df.groupby(['도시', '구', '동']).sum().reset_index()
parking_df.reset_index(inplace=True, drop=True)

#merge
train_df = pd.merge(train_df, light_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, child_area_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, parking_df, how='left', on=['도시', '구', '동'])

test_df = pd.merge(test_df, light_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, child_area_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, parking_df, how='left', on=['도시', '구', '동'])

test_x = test_df.drop(columns=['ID']).copy()
train_x = train_df[test_x.columns].copy()
train_y = train_df['ECLO'].copy()

categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)
for i in categorical_features:
    le = LabelEncoder()
    le=le.fit(train_x[i]) 
    train_x[i]=le.transform(train_x[i])
    test_x[i]=le.transform(test_x[i])

train_x.fillna(0, inplace=True)
test_x.fillna(0, inplace=True)

class CustomDataset(Dataset):
    def __init__(self, is_split=False):
        super(CustomDataset, self).__init__()
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
    
class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.x = test_x
        # 텐서 변환
        self.x = torch.tensor(self.x.values).float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        return x
    

    


