import torch
import torch.nn as nn

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
    
    def get_weights(self):
        """Extract model weights as a list of tensors"""
        return [param.data for param in self.parameters()]
    
    def set_weights(self, weights):
        """Set model weights from a list of tensors"""
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.copy_(weight) 