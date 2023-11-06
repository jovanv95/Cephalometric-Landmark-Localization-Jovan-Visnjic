import torch
import torch.nn as nn
import torch.nn.functional as F





class StackedHourglass(nn.Module):

    class Hourglass(nn.Module):

        class Residual(nn.Module):
            def __init__(self,input,output):
                super(StackedHourglass.Hourglass.Residual, self).__init__()

                self.conv1 = nn.Conv2d(input, input, kernel_size=1)
                self.conv2 = nn.Conv2d(input, input, kernel_size=3,padding=1)
                self.conv3 = nn.Conv2d(input, output, kernel_size=1)
                self.relu = nn.LeakyReLU()

                if input != output:
                    self.projection = nn.Conv2d(input, output, kernel_size=1)
                else:
                    self.projection = None


            def forward(self,x):
                identity = x
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.relu(x)

                if self.projection is not None:
                    identity = self.projection(identity)
                x = self.relu(x + identity)

                return x

        def __init__(self,keypoint_number = 22):
            super(StackedHourglass.Hourglass, self).__init__()
            self.dropout = nn.Dropout2d(p=0.1)
            self.maxpool = nn.MaxPool2d(2, stride=2)
            self.Residual1 = self.Residual(64,128)
            self.Residual2 = self.Residual(128,256)
            self.Residual3 = self.Residual(256,512)
            self.Residual4 = self.Residual(512,512)
            self.ResidualMid1 = self.Residual(512,512)
            self.ResidualMid2 = self.Residual(512,512)
            self.ResidualMid3 = self.Residual(512,512)
            self.ConvTrans1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            self.Residual5 = self.Residual(512,512)
            self.Residual6 = self.Residual(512,256)
            self.ConvTrans2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
            self.Residual7 = self.Residual(256,128)
            self.ConvTrans3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
            self.Residual8 = self.Residual(128,64)
            self.ConvTrans4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
            self.ConvSeondLast = nn.ConvTranspose2d(64, 256, kernel_size=1, stride=1)
            self.ConvLast = nn.ConvTranspose2d(256, keypoint_number, kernel_size=1, stride=1)
            self.fc = nn.Linear(360448,22*2)
            self.batch64 = nn.BatchNorm2d(64)
            self.batch128 = nn.BatchNorm2d(128)
            self.batch256 = nn.BatchNorm2d(256)
            self.batch64_2 = nn.BatchNorm2d(64)
            self.batch128_2 = nn.BatchNorm2d(128)
            self.batch256_2 = nn.BatchNorm2d(256)
            self.batch256_2 = nn.BatchNorm2d(256)
            self.batch512 = nn.BatchNorm2d(512)
            self.batch512_1 = nn.BatchNorm2d(512)
            self.batch512_2 = nn.BatchNorm2d(512)
            self.batch512_3 = nn.BatchNorm2d(512)
            self.batch512_4 = nn.BatchNorm2d(512)
            
            

        def forward(self, x):
            x = self.Residual1(x)
            x = self.batch128(x)
            x = self.dropout(x)
            x1 = x 
            x = self.maxpool(x)
            x = self.Residual2(x)
            x = self.batch256(x)
            x2 = x 
            x = self.maxpool(x)
            x = self.Residual3(x)
            x = self.batch512(x)
            x = self.dropout(x)
            x3 = x 
            x = self.maxpool(x)
            x = self.Residual4(x)
            x = self.batch512_1(x)
            x = self.dropout(x)
            x4 = x
            x = self.maxpool(x)
            x = self.ResidualMid1(x)
            x = self.batch512_2(x)
            x = self.dropout(x)
            x = self.ResidualMid2(x)
            x = self.batch512_3(x)
            x = self.dropout(x)
            x = self.ResidualMid3(x)
            x = self.batch512_4(x)
            x = self.dropout(x)
            x = self.ConvTrans1(x)
            x += x4
            x = self.Residual5(x)
            x = self.dropout(x)
            x = self.ConvTrans2(x)
            x += x3
            x = self.Residual6(x)
            x = self.batch256_2(x)
            x = self.dropout(x)
            x = self.ConvTrans3(x)
            x += x2
            x = self.Residual7(x)
            x = self.batch128_2(x)
            x = self.dropout(x)
            x = self.ConvTrans4(x)
            x += x1
            x = self.Residual8(x)
            x = self.batch64_2(x)
            x = self.dropout(x)
            img = x
            heatmap = self.ConvSeondLast(x)
            heatmap = self.ConvLast(heatmap)
            heatmap = F.interpolate(heatmap, scale_factor=2, mode='bilinear', align_corners=False)

            return img , heatmap

    def __init__(self):
        super(StackedHourglass, self).__init__() 
        self.initialize_weights()
        self.sigmoid = nn.Sigmoid()
        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64)
        )
        
        self.h1 = self.Hourglass(22)
        self.h2 = self.Hourglass(22)
        self.h3 = self.Hourglass(22)
        #self.h4 = self.Hourglass(22)
        #self.h5 = self.Hourglass(22)
        #self.h6 = self.Hourglass(22)
        #self.h7 = self.Hourglass(22)
        #self.h8 = self.Hourglass(22)
    
    def forward(self,x):

        if torch.isnan(x).any():
            print("original")
        y = x
        
        x = self.first_layer(x)
        if torch.isnan(x).any():
            print("first_layer")
            torch.save(y, 'tensor.pt')
            raise KeyboardInterrupt
        
        x,heatmap = self.h1(x)
        x,heatmap = self.h2(x)
        x,heatmap = self.h3(x)
        heatmap = self.sigmoid(heatmap)

        return heatmap
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)






