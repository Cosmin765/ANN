#################### 1 [imports] ####################
import json
import time
import psutil
import collections
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from typing import Optional, Callable
import os
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
import socket
from tqdm import tqdm
#################### 1 ####################

#################### 2 [init] ####################
LOCAL = socket.gethostname().startswith('ctu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cudnn.benchmark = True
pin_memory = True
enable_half = device == torch.device('cuda')  # Disable for CPU, it is slower!
scaler = GradScaler(device.type, enabled=enable_half)
#################### 2 ####################


#################### 3 ####################
class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        # Runtime transforms are not implemented in this simple cached dataset.
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
#################### 3 ####################


#################### 4 ####################
class CIFAR100_noisy_fine(Dataset):
    """
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    """

    def __init__(
        self, root: str, train: bool, transform: Optional[Callable], download: bool
    ):
        cifar100 = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]
#################### 4 ####################


#################### 5 [load data] ####################
class GroupedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.grouped_indices = self.group_by_label()

    def group_by_label(self):
        from collections import defaultdict

        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset):
            label_to_indices[label].append(idx)

        grouped_indices = []
        for indices in label_to_indices.values():
            grouped_indices.extend(indices)

        return grouped_indices

    def __len__(self):
        return len(self.grouped_indices)

    def __getitem__(self, idx):
        original_idx = self.grouped_indices[idx]
        return self.dataset[original_idx]


base_transforms = [
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.50707585, 0.48655054, 0.4409193), std=(0.20089693, 0.19844234, 0.20229685), inplace=True),
    # TODO: invert first and third item
]

transform = v2.Compose(base_transforms)

applied_transforms = [
    transform,
    v2.Compose([v2.RandomHorizontalFlip(p=1.0), *base_transforms]),
    v2.Compose([v2.RandomResizedCrop(size=(32, 32), scale=(0.8, 0.8)), *base_transforms]),
    v2.Compose([v2.RandomPerspective(distortion_scale=0.4, p=1.0), *base_transforms]),
    v2.Compose([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), *base_transforms]),
    v2.Compose([v2.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3)), *base_transforms]),
    v2.Compose([v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10), *base_transforms]),
]

base_train_set = CIFAR100_noisy_fine('/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100', download=False, train=True, transform=None)
base_test_set = CIFAR100_noisy_fine('/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100', download=False, train=False, transform=transform)

images = base_train_set.data
labels = base_train_set.targets

train_set = torch.utils.data.ConcatDataset([SimpleCachedDataset([(applied_transform(img), label)
                                                                 for img, label in zip(images, labels)])
                                            for applied_transform in applied_transforms])
train_set = GroupedDataset(train_set)
test_set = SimpleCachedDataset(base_test_set)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True, pin_memory=pin_memory)
test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)
#################### 5 ####################

#################### 6 [load model] ####################
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            # Block 3
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
            nn.Dropout(0.4),
            # Block 4
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
            nn.Dropout(0.4),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

model = VGG16()
# model = timm.create_model("hf_hub:grodino/resnet18_cifar10", pretrained=True)

# for name, param in model.named_parameters():
#     # if 'layer4' not in name:
#     param.requires_grad = False  # freeze parameters

# for param in model.parameters():
#     param.requires_grad = False  # freeze parameters

# model.fc = nn.Linear(512, 100)
# nn.init.kaiming_uniform_(model.fc.weight)
model = model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.001, fused=True)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * 5)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-2, total_steps=len(train_loader) * 2)
#################### 6 ####################


#################### 7 [train] ####################
cutmix = v2.CutMix(num_classes=100, alpha=1.5)
mixup = v2.MixUp(num_classes=100, alpha=0.8)


def apply_augmentation(images, targets):
    return cutmix(images, targets)
    # return mixup(images, targets)
    # if torch.rand(1).item() < 0.5:
    #     return cutmix(images, targets)
    # else:
    #     return mixup(images, targets)


def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)  # targets: (64,)
        # inputs, targets = cutmix(inputs, targets)  # targets: (64, 100)
        # inputs, targets = mixup(inputs, targets)  # targets: (64, 100)
        inputs, targets = apply_augmentation(inputs, targets)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)  # outputs: (64, 100)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(axis=1)

        total += targets.size(0)
        correct += predicted.eq(targets.argmax(axis=1)).sum().item()
        # correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total
#################### 7 ####################


#################### 8 [validate] ####################
@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total, val_loss
#################### 8 ####################


#################### 9 [final benchmark] ####################
@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels
#################### 9 ####################


#################### 10 [iterate] ####################
metrics = collections.defaultdict(list)

best = 0.0
epochs = list(range(60))
warmed_up = False
warm_up_threshold = 10
epochs = list(range(-warm_up_threshold, 0)) + epochs
last_val_acc = 0

with tqdm(epochs) as tbar:
    for epoch in tbar:
        # if epoch >= 0 and not warmed_up:
        #     for param in model.parameters():
        #         param.requires_grad = True  # unfreeze parameters
        #     warmed_up = True
        #     model.load_state_dict(torch.load("model_local.pt", weights_only=True, map_location=device))

        train_acc = train()
        val_acc, val_loss = val()

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_acc > best:
            if best > 0:
                print("Saved model")
                torch.save(model.state_dict(), "model_local.pt")
            best = val_acc
        # elif val_acc < best and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #     print("Switched to ReduceLROnPlateau")
        #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-6)
        print(f"[{epoch}] Train: {train_acc:.2f}, Val: {val_acc:.2f}")
        tbar.set_description(
            f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Delta val: {val_acc - last_val_acc}, Best: {best:.2f}, Epoch: {epoch}")
        last_val_acc = val_acc

        metrics['memory'].append(psutil.virtual_memory().used)
        metrics['train_accuracy'].append(float(train_acc))
        metrics['val_accuracy'].append(float(val_acc))
        metrics['val_loss'].append(float(val_loss))
        metrics['timestamp'].append(time.time())

model.load_state_dict(torch.load("model_local.pt", weights_only=True, map_location=device))
#################### 10 ####################

#################### 11 [results] ####################
with open('metrics.json', 'w') as fd:
    json.dump(metrics, fd, indent=4)

data = {
    "ID": [],
    "target": []
}


for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
if LOCAL:
    os.makedirs('results', exist_ok=True)
    df.to_csv(os.path.join('results', 'submission.csv'), index=False)
else:
    df.to_csv("/kaggle/working/submission.csv", index=False)
#################### 11 ####################
