import torch
import torch.nn as nn

class EmployeeClassifier(nn.Module):
    def __init__(self,input_size):
        super().__init__()

        self.network=nn.Sequential(
            nn.Linear(input_size,16),
            nn.ReLU(),

            nn.Linear(16,8),
            nn.ReLU(),

            nn.Linear(8,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.network(x)