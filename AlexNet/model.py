import torch
import torch.nn as nn

# alexnet
class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.conv = nn.Sequential(
            # (B, 1, 32,32)
            # layer1
            nn.Conv2d(1, 8, 5), # B, 8, 28,28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1), # B, 8, 24,24
            # layer2
            nn.Conv2d(8, 16, 5), # B, 16, 20,20
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4), # B, 128, 5,5
            # layer3
            nn.Conv2d(16, 32, 3, padding=1), # B, 32, 3,3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3), # B, 32, 1,1
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 1 * 1, 64),  # B, 32
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 10), # B, 10
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        # x = self.avgpool(x)
        # print(x.shape)
        x = x.view(-1, 32*1*1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # print("1:", x[:3])
        # print("2:", x[:3])
        # x = torch.softmax(x, dim=1) # 64, 10
        # print("3:", x1[:3])
        # x = torch.softmax(x, dim=1) # 64, 10
        # print("3:", x[:3])
        # print(x.shape)
        return x
