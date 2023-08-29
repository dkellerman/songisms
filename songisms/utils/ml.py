import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        distance_positive = F.pairwise_distance(anchor, pos, 2)
        distance_negative = F.pairwise_distance(anchor, neg, 2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class SiameseTripletNetwork(nn.Module):
    def __init__(self):
        super(SiameseTripletNetwork, self).__init__()
        self.fc1 = nn.Linear(336, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 256)

    def forward_once(self, x):
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        z = self.out(z)
        return z

    def forward(self, anchor, pos=None, neg=None):
        if pos is None and neg is None:
            return self.forward_once(anchor)

        anchor_out = self.forward_once(anchor)
        pos_out = self.forward_once(pos)
        neg_out = self.forward_once(neg)
        return anchor_out, pos_out, neg_out
