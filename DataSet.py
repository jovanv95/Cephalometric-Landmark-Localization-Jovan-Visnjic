import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch




class Xray_DataSet(Dataset):
    def __init__(self, path2data,df_x,df_y, transform,mean,std):
        self.mean = mean
        self.std = std
        self.path2data = path2data
        self.x = df_x
        self.y = df_y
        self.ids=self.x.index
        self.transform = transform
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        index = self.ids[idx]
        image = Image.open(f"{self.path2data}/{index}.jpg")
        x = self.x.iloc[index]
        y = self.y.iloc[index]
        
        image,keypoints = self.transform(image,x,y,mean = self.mean,std = self.std)

        return image, keypoints


