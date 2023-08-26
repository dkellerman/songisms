import torch
import torch.nn
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class SiameseTripletNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseTripletNetwork, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, 5)  # chnl-in, out, krnl
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(1024, 512)   # [64*4*4, x]
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 2)     # n values
        self.pool1 = torch.nn.MaxPool2d(2, stride=2) # kernel, stride
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)
        self.drop1 = torch.nn.Dropout(0.25)
        self.drop2 = torch.nn.Dropout(0.50)

    def forward_once(self, x):
        # convolution phase                 # x is [bs, 1, 28, 28]
        z = torch.relu(self.conv1(x))       # Size([bs, 32, 24, 24])
        z = self.pool1(z)                   # Size([bs, 32, 12, 12])
        z = self.drop1(z)
        z = torch.relu(self.conv2(z))       # Size([bs, 64, 8, 8])
        z = self.pool2(z)                   # Size([bs, 64, 4, 4])

        # neural network phase
        z = z.reshape(-1, 1024)             # Size([bs, 1024])
        z = torch.relu(self.fc1(z))         # Size([bs, 512])
        z = self.drop2(z)
        z = torch.relu(self.fc2(z))         # Size([bs, 256])
        z = self.fc3(z)                     # Size([bs, n])
        return z


    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output
