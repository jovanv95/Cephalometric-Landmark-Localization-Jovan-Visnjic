from model import *
from data_load import *
from train import *
from torch.nn import MSELoss
from loss import HourGlassLoss
from param import *

train(
    epoch_num = epoch_num,
    model = model,
    train_loader = train_dl,
    test_loader =test_dl,
    optimizer = optimizer,
    criterion =HourGlassLoss(22,1),
    scheduler=scheduler,
    )
