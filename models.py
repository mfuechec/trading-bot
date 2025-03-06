import torch
import torch.nn as nn

class SimpleDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_size)
        
        # L2 regularization will be applied through weight_decay in optimizer
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        return self.fc2(x)

class ComplexDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x) 