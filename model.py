import torch
from torch import nn
import numpy as np

class Pollen(nn.Module):
    
    def __init__(self):
        super(Pollen, self).__init__()
        
        self.conv11 = nn.Conv2d(1, 10, (5,5))
        self.conv12 = nn.Conv2d(10, 20, (3,3))
        self.conv13 = nn.Conv2d(20, 100, (3,3))
        self.conv14 = nn.Conv2d(100, 200, (3,3))
        
        self.conv21 = nn.Conv2d(1, 70, (1,7))
        self.conv22 = nn.Conv2d(70, 140, (1,5))
        self.conv23 = nn.Conv2d(140, 200, (3,3))
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.bn11 = nn.BatchNorm2d(10)
        self.bn12 = nn.BatchNorm2d(20)
        self.bn13 = nn.BatchNorm2d(100)
        self.bn14 = nn.BatchNorm2d(200)
        
        self.bn31 = nn.BatchNorm2d(70)
        self.bn32 = nn.BatchNorm2d(140)
        self.bn33 = nn.BatchNorm2d(200)
        
        self.pad1 = nn.ReplicationPad2d(2)
        self.pad2 = nn.ReplicationPad2d(1) 
        
        self.pad31 = nn.ReplicationPad2d((3, 3, 0, 0))
        self.pad32 = nn.ReplicationPad2d((2, 2, 0, 0))
        
        self.fc1 = nn.Linear(5200, 400)
        self.fc2 = nn.Linear(1600, 120)
        self.fc3 = nn.Linear(1200, 100) 
        self.fc4 = nn.Linear(620, 8)
                
    def forward(self, x):

        x0 = np.uint8(x[0])
        x0_mean = x0.mean()
        x0 = torch.tensor(x[0], dtype=float)
        x1 = torch.tensor(x[1]+np.full(x[1].shape, x0_mean), dtype=float)
        x2 = torch.tensor(x[2]+np.full(x[2].shape, x0_mean), dtype=float) 
        x0 = x0.unsqueeze(1)
        x0 = self.pad1(x0)
        x0 = x0.double()
        x0 = self.conv11(x0)
        x0 = self.relu(x0)
        x0 = self.bn11(x0)
        x0 = self.pool(x0)
        x0 = self.pad2(x0)
        x0 = self.conv12(x0)
        x0 = self.relu(x0)
        x0 = self.bn12(x0)
        x0 = self.pool(x0)
        x0 = self.conv13(x0)
        x0 = self.relu(x0)
        x0 = self.bn13(x0)
        x0 = self.conv14(x0)
        x0 = self.relu(x0)
        x0 = self.bn14(x0)
        x0 = x0.view(x0.size(0), 1, -1)
        x0 = self.fc1(x0)
        
        x1 = x1.unsqueeze(1)
        x1 = self.pad31(x1)
        x1 = self.conv21(x1)
        x1 = self.relu(x1)
        x1 = self.bn31(x1) 
        x1 = self.pad32(x1) 
        x1 = self.conv22(x1)
        x1 = self.relu(x1)
        x1 = self.bn32(x1) 
        x1 = self.pool(x1)
        x1 = self.pad2(x1)
        x1 = self.conv33(x1) 
        x1 = self.relu(x1)
        x1 = self.bn33(x1) 
        x1 = self.pool(x1)
        x1 = x1.view(x1.size(0), 1, -1)
        x1 = self.fc2(x1)
        
        x2 = x2.unsqueeze(1)
        x2 = self.pad31(x2) 
        x2 = self.conv31(x2) 
        x2 = self.relu(x2)
        x2 = self.bn31(x2) 
        x2 = self.pad32(x2) 
        x2 = self.conv32(x2) 
        x2 = self.relu(x2)
        x2 = self.bn32(x2) 
        x2 = self.pool(x2)
        x2 = self.pad2(x2)
        x2 = self.conv33(x2) 
        x2 = self.relu(x2)
        x2 = self.bn33(x2)
        x2 = self.pool(x2)
        x2 = x2.view(x2.size(0), 1, -1)
        x2 = self.fc3(x2)
        
        y = torch.cat((x0, x1, x2), dim=2)
        y = self.fc4(y)
        
        return y

