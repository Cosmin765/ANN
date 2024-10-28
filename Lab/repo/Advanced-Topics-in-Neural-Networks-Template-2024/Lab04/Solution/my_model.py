import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageTransformationPredictor(nn.Module):
    def __init__(self):
        super(ImageTransformationPredictor, self).__init__()

        self.image1_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dtype=torch.float),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dtype=torch.float),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.image2_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dtype=torch.float),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dtype=torch.float),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dtype=torch.float),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.months_fc = nn.Linear(1, 64, dtype=torch.float)

        self.fc1 = nn.Linear(256 * 28 * 28 + 256 * 28 * 28 + 64, 512, dtype=torch.float)
        self.fc2 = nn.Linear(512, 3 * 224 * 224, dtype=torch.float)

    def forward(self, image1, image2, diff_months):
        img1_features = self.image1_conv(image1)
        img1_features = img1_features.view(img1_features.size(0), -1)

        img2_features = self.image2_conv(image2)
        img2_features = img2_features.view(img2_features.size(0), -1)

        diff_months = diff_months.view(-1, 1)
        months_features = F.relu(self.months_fc(diff_months))

        combined = torch.cat([img1_features, img2_features, months_features], dim=1)

        x = F.relu(self.fc1(combined))
        output = self.fc2(x)

        output = output.view(-1, 3, 224, 224)
        return output
