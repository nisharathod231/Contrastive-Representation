import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqFwd(nn.Module):
    def __init__(self, inp_dim, out_dim, stride=1):
        super(SeqFwd, self).__init__()
        self.conv1 = nn.Conv2d(inp_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp_dim != out_dim:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(inp_dim, out_dim, kernel_size=2, stride=stride, bias=False),
                nn.Conv2d(inp_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, z_dim, block=SeqFwd, num_blocks = [2,2,2,2], num_classes=10):
        super(Encoder, self).__init__()
        self.inp_dim = 64
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  #isko 32x32 rakhna
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output is 16x16 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) #64x64
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.Enco_help(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.Enco_help(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.Enco_help(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.Enco_help(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, z_dim)
    def Enco_help(self, block, out_dim, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inp_dim, out_dim, stride))
            self.inp_dim = out_dim
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class Classifier(nn.Module):
    def __init__(self, encoded_dim, classes = 10):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(encoded_dim, classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        layer1_output = self.fc(X)
        layer1_output = self.softmax(layer1_output)
        return layer1_output