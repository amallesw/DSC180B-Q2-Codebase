import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionClassifierNN(nn.Module):
    def __init__(self, input_size=512, num_classes=7, dropout_rate=0.5):
        
        super(EmotionClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first FC layer
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second FC layer
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize layers' weights
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Applying dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Applying dropout after activation
        x = self.fc3(x)
        return x
