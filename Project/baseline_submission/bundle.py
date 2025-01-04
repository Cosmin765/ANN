#################### 1 [imports] ####################
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
from typing import Optional, Callable
import os
import timm
import numpy as np
import pandas as pd
from torchvision.transforms import v2
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
import socket
from tqdm import tqdm
import torch.nn.functional as F
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
        self, root: str, train: bool, transform: Optional[Callable], download: bool, override_noise=False
    ):
        cifar100 = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        # if train and not override_noise:
        if train:
            noisy_label_file = os.path.join(root, "CIFAR-100-noisy.npz")
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} need {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")

            old_targets = targets
            targets = noise_file["noisy_label"]

            matching = [a == b for a, b in zip(old_targets, targets)]
            percent = len([a for a in matching if not a]) / len(matching)
            print(percent)

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]
#################### 4 ####################


#################### 5 [load data] ####################
# DONE: idea - when rotating, make the black pixels have the mean value of the pixels
# DONE: idea - group instances with the same label
# DONE: idea - do the loss_threshold but with a much smaller batch size
# DONE: idea - disable loss_threshold when the global_min_loss becomes small enough
# TODO: idea - perform some kind of similarity to take out the noisy labels

class RandomRotationFillBlackCorners(torch.nn.Module):
    def __init__(self, degrees):
        super(RandomRotationFillBlackCorners, self).__init__()
        if isinstance(degrees, tuple):
            self.degrees = (degrees[0], degrees[1])
        else:
            self.degrees = (-degrees, degrees)

    def forward(self, image):
        rand_degrees = torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        image_array = np.array(image)
        mean_color = tuple(np.mean(image_array.reshape(-1, image_array.shape[-1]), axis=0).astype(int))
        return image.rotate(rand_degrees, expand=False, fillcolor=mean_color)


train_transform_v2 = v2.Compose([
    # v2.RandomCrop(32, padding=4, pad_if_needed=True, padding_mode='edge'),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomRotationFillBlackCorners(degrees=15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.50707585, 0.48655054, 0.4409193), std=(0.20089693, 0.19844234, 0.20229685), inplace=True)
])

train_transform_v3 = v2.Compose([
    v2.RandomResizedCrop((32, 32)),
    v2.RandomHorizontalFlip(p=0.75),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomVerticalFlip(p=0.75),
    RandomRotationFillBlackCorners(degrees=15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.50707585, 0.48655054, 0.4409193), std=(0.20089693, 0.19844234, 0.20229685), inplace=True)
])

train_transform_v4 = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    v2.RandomRotation(15),
    v2.ToTensor(),
    v2.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

val_transforms_v1 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.50707585, 0.48655054, 0.4409193), std=(0.20089693, 0.19844234, 0.20229685), inplace=True)
])

basic_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_transform = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    RandomRotationFillBlackCorners(degrees=15),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    v2.ToTensor(),
    v2.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])
val_transforms = val_transforms_v1


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


if LOCAL:
    os.makedirs('dataset', exist_ok=True)
    train_set = CIFAR100_noisy_fine(os.path.join('dataset', 'fii-atnn-2024-project-noisy-cifar-100'),
                                    download=False, train=True, transform=train_transform, override_noise=True)
    test_set = CIFAR100_noisy_fine(os.path.join('dataset', 'fii-atnn-2024-project-noisy-cifar-100'),
                                   download=False, train=False, transform=val_transforms)

    train_set_not_grouped = train_set
    train_set = GroupedDataset(train_set)

    import json
    from torchvision.transforms.functional import to_pil_image
    import cv2
    with open('meta.json') as fd:
        labels = json.load(fd)['fine_label_names']

    # for image_np, label_index in train_set:
    #     label = labels[label_index]
    #     image = np.array(to_pil_image(image_np))
    #     height, width = image.shape[:2]
    #     print(label)
    #
    #     cv2.imshow('imagine', cv2.resize(image, (width * 16, height * 16)))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
else:
    train_set = CIFAR100_noisy_fine(
        '/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100', download=False,
        train=True, transform=train_transform)
    test_set = CIFAR100_noisy_fine(
        '/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100', download=False,
        train=False, transform=val_transforms)

    # train_set = GroupedDataset(train_set)

train_set = SimpleCachedDataset(train_set)
test_set = SimpleCachedDataset(test_set)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=pin_memory)
if LOCAL:
    train_set_no_transform = CIFAR100_noisy_fine(os.path.join('dataset', 'fii-atnn-2024-project-noisy-cifar-100'),
                                                 download=False, train=True, transform=basic_transforms, override_noise=True)
    train_val_set = SimpleCachedDataset([(i, input, target) for i, (input, target) in enumerate(train_set_not_grouped)])
    train_val_loader = DataLoader(train_val_set, batch_size=1, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=pin_memory)
else:
    test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)
#################### 5 ####################

#################### 6 [load model] ####################
model = timm.create_model("hf_hub:grodino/resnet18_cifar10", pretrained=True)

for name, param in model.named_parameters():
    if 'layer4' not in name:
        param.requires_grad = False  # freeze parameters

# for param in model.parameters():
#     param.requires_grad = False  # freeze parameters

model.fc = nn.Linear(512, 100)
nn.init.kaiming_uniform_(model.fc.weight)
model = model.to(device)
# if os.path.isfile("model.pt"):
# model.load_state_dict(torch.load("model_local.pt", weights_only=True, map_location=device))
# model = torch.jit.script(model)  # does not work for this model
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.001, fused=True)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * 20)
#################### 6 ####################


#################### 7 [train] ####################
def train():
    model.train()
    correct = 0
    total = 0

    # cutmix = v2.CutMix(num_classes=100, alpha=1.0)
    mixup = v2.MixUp(num_classes=100, alpha=0.2)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)  # targets: (64,)
        inputs, targets = mixup(inputs, targets)  # targets: (64, 100)
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
        scheduler.step()

    return 100.0 * correct / total
#################### 7 ####################


#################### 7.5 [train val] ####################
def train_val():
    # manually validate the training predictions

    if not LOCAL:
        return

    model.eval()
    correct = 0
    total = 0

    clean_confidences_correct = []
    clean_confidences_wrong = []
    noise_confidences_correct = []
    noise_confidences_wrong = []

    for indices, inputs, targets in train_val_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)  # targets: (64,)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)  # outputs: (64,)

        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, dim=1)

        for index, target, pred, confidence in zip(indices, targets, predicted, confidences):
            target_label = labels[target]
            pred_label = labels[pred]

            input, clean_target = train_set_no_transform[index]
            clean_label = labels[clean_target]
            image = np.array(to_pil_image(input))
            height, width = image.shape[:2]

            if target_label == clean_label:
                if pred_label == target_label:
                    clean_confidences_correct.append(confidence.item())
                else:
                    clean_confidences_wrong.append(confidence.item())
            else:
                if pred_label == target_label:
                    noise_confidences_correct.append(confidence.item())
                else:
                    noise_confidences_wrong.append(confidence.item())

            # most likely clean?
            # print(f"{index}. Predicted: {pred_label}, Target: {target_label}, Clean: {clean_label}, Confidence: {confidence}")
            # cv2.imshow('val', cv2.resize(image, (width * 16, height * 16)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    clean_confidences_correct = np.array(clean_confidences_correct)  # 0.23411131907512658
    clean_confidences_wrong = np.array(clean_confidences_wrong)  # 0.10773168840667008
    noise_confidences_correct = np.array(noise_confidences_correct)  # 0.16427118599572035
    noise_confidences_wrong = np.array(noise_confidences_wrong)  # 0.10612681301449864

    clean_confidences_correct_mean = np.mean(clean_confidences_correct)
    clean_confidences_wrong_mean = np.mean(clean_confidences_wrong)
    noise_confidences_correct_mean = np.mean(noise_confidences_correct)
    noise_confidences_wrong_mean = np.mean(noise_confidences_wrong)

    print((len(noise_confidences_correct) + len(noise_confidences_wrong)) / (len(noise_confidences_correct) + len(noise_confidences_wrong) + len(clean_confidences_correct) + len(clean_confidences_wrong)))

    return 100.0 * correct / total


#################### 8 [validate] ####################
@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total
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
best = 0.0
epochs = list(range(20))
warmed_up = False
warm_up_threshold = min(len(epochs) // 4, 5)
epochs = list(range(-warm_up_threshold, 0)) + epochs
last_val_acc = 0

with tqdm(epochs) as tbar:
    for epoch in tbar:
        if epoch >= 0 and not warmed_up:
            for param in model.parameters():
                param.requires_grad = True  # unfreeze parameters
            warmed_up = True

        # if LOCAL:
        #     train_acc = train_val()  # skip training locally
        #     continue

        train_acc = train()
        val_acc = val()
        if val_acc > best:
            if best > 0:
                print("Saved model")
                torch.save(model.state_dict(), "model_local.pt")
            best = val_acc
        tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Delta val: {val_acc - last_val_acc}, Best: {best:.2f}, Epoch: {epoch}")
        last_val_acc = val_acc
#################### 10 ####################

#################### 11 [results] ####################
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
