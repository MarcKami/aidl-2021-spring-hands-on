import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.pad = nn.ConstantPad2d(2,0)
        self.conv1 = ConvBlock(1,6,5)
        self.conv2 = ConvBlock(6,16,5)
        self.conv3 = ConvBlock(16,32,5)
        self.mlp = nn.Sequential( nn.Linear(800, 240),
                            nn.ReLU(),
                            nn.Linear(240,84),
                            nn.ReLU(),
                            nn.Linear(84,15),
                            nn.LogSoftmax(dim=1))
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Obtain the parameters of the tensor in terms of:
        # 1) batch size
        # 2) number of channels
        # 3) spatial "height"
        # 4) spatial "width"
        bsz, nch, height, width = x.shape  
        # Flatten the feature map with the view() operator 
        # within each batch sample  
        x = x.view(x.shape[0],-1)
        y = self.mlp(x)
        return y


class ConvBlock(nn.Module):

    def __init__(self, num_inp_channels, num_out_fmaps, kernel_size, pool_size=2):
        super().__init__()
        #TODO: define the 3 modules needed
        self.conv = nn.Conv2d(num_inp_channels,num_out_fmaps,kernel_size)
        self.maxpool = nn.MaxPool2d(pool_size)
        self.relu = nn.ReLU()
  
    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))
