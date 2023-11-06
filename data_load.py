import imp
from DataSet import *
from trans import *
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from param import *

df_x_train = df_x_train.fillna(df_x_train.mean())
df_y_train = df_y_train.fillna(df_y_train.mean())
df_x_test = df_x_test.fillna(df_x_test.mean())
df_y_test = df_y_test.fillna(df_y_test.mean())





ds = Xray_DataSet(path_train,df_x_train,df_y_train,transformer_mean_std,1,1)

def get_mean_std(dataset):

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        mean = 0.
        std = 0.
        num_samples = 0.
        for x , _ in data_loader:
                batch_samples = x.size(0)
                x = x.view(batch_samples, x.size(1), -1)
                mean += x.mean(2).sum(0)
                std += x.std(2).sum(0)
                num_samples += batch_samples

        mean /= num_samples
        std /= num_samples  
        mean = torch.mean(mean) 
        std = torch.mean(std) 

        return mean , std


mean ,std = get_mean_std(ds)






dataset_train = Xray_DataSet(path_train,df_x_train,df_y_train,transformer_train,mean,std)
dataset_test = Xray_DataSet(path_test,df_x_test,df_y_test,transformer_test,mean,std)





train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)










