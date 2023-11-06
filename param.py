import pandas as pd
from model import StackedHourglass
from torch.optim.lr_scheduler import StepLR
import torch

path_train = "../full_dataset/full/split/train"
path_test = "../full_dataset/full/split/test"

df_x_train = pd.read_csv("../full_dataset/full/split/x_train.csv",index_col=0)
df_y_train = pd.read_csv("../full_dataset/full/split/y_train.csv",index_col=0)
df_x_test = pd.read_csv("../full_dataset/full/split/x_test.csv",index_col=0)
df_y_test = pd.read_csv("../full_dataset/full/split/y_test.csv",index_col=0)

batch_size = 1
keypoint_number = 22

lr = 0.00025
weight_decay = 0.09
epoch_num = 1000

device = "cuda:0"

model = StackedHourglass().to("cuda:0")
optimizer = torch.optim.Rprop(model.parameters(),lr=lr)
scheduler = StepLR(optimizer, step_size=200, gamma=0.5,verbose= True)