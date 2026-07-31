"""
Simple CNN model for Fake Currency Detection (Real vs Fake).

Kept intentionally small (3 conv blocks + 2 fully-connected layers) so it
trains fast on a modest dataset and a CPU/laptop GPU, while still being
deep enough to learn texture/pattern differences between real and fake notes.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, img_size: int = 128):
        super(SimpleCNN, self).__init__()

        # ---- Conv Block 1 ----
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # ---- Conv Block 2 ----
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # ---- Conv Block 3 ----
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves H and W each time

        # After 3 pooling layers: img_size / (2*2*2) = img_size / 8
        reduced_size = img_size // 8
        flattened_size = 64 * reduced_size * reduced_size

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))   # -> 16 x H/2 x W/2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))   # -> 32 x H/4 x W/4
        x = self.pool(self.relu(self.bn3(self.conv3(x))))   # -> 64 x H/8 x W/8

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # raw logits (use CrossEntropyLoss, which applies softmax internally)
        return x


if __name__ == "__main__":
    # Quick sanity check: run a dummy batch through the model
    model = SimpleCNN(num_classes=2, img_size=128)
    dummy_input = torch.randn(4, 3, 128, 128)  # batch of 4 RGB images
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expected: torch.Size([4, 2])
