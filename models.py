import torch
from torch import nn


class ClassifierMLP(nn.Module):
    def __init__(self, in_feature, num_cls):
        super(ClassifierMLP, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_cls),
        )
    def forward(self, x):
        return self.classify(x)
    
################################################################################    
class ClassifierCNN(nn.Module):
    def __init__(self, num_cls):
        super(ClassifierCNN, self).__init__()
        self.num_cls = num_cls
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=5, bias=False), # 16x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.dense=nn.Sequential(
            nn.Linear(32*9*9, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_cls)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32*9*9)
        return self.dense(x)
        
################################################################################    
class ClassifierCNN2(nn.Module):
    def __init__(self, num_cls):
        super(ClassifierCNN2, self).__init__()
        self.num_cls = num_cls
        self.classify = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=5, bias=False), # 16x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # 4x4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_cls, kernel_size=4, stride=2, padding=0, bias=False), # 1,1
            )

    def forward(self, x):
        x = self.classify(x)
        return x.view(-1,self.num_cls)