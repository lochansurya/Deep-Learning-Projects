import torch
import torch.nn as nn
import torch.nn.functional as F





class Encoder(nn.Module):
    def __init__(self, output_dim):
        super(Encoder, self).__init__()
        # Define the custom architecture
        self.encoder = nn.Sequential(
            # Convolutional layers with batch normalization
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Linear layer to transform features to the desired output dimension
        self.fc = nn.Linear(512 * 7 * 7, output_dim)

    def forward(self, x):
        # Forward pass through the encoder layers
        x = self.encoder(x)
        # Forward pass through the adaptive average pooling layer
        x = self.avgpool(x)
        # Flatten the output
        x = torch.flatten(x, 1)
        # Forward pass through the linear layer
        x = self.fc(x)
        return x

    

class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes=10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, num_classes)

    def forward(self, x):
        return self.fc1(x)

# class Classifier(nn.Module):
#     def __init__(self, z_dim, num_classes=10):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(z_dim, 512)  # First fully connected layer
#         self.relu = nn.ReLU(inplace=True)  # ReLU activation function
#         self.fc2 = nn.Linear(512, num_classes)  # Second fully connected layer

#     def forward(self, x):
#         x = self.fc1(x)  # Forward pass through the first fully connected layer
#         x = self.relu(x)  # Apply ReLU activation function
#         x = self.fc2(x)  # Forward pass through the second fully connected layer
#         return x