import os.path

import torch
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch import optim
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import wandb
from augmentations import augmentations
import argparse
from early_stopping import EarlyStopping

DATASET_PATH = './dataset'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
criterion = nn.CrossEntropyLoss()
pin_memory = True
enable_half = device != torch.device('cpu')  # Disable for CPU, it is slower!
scaler = torch.GradScaler(device.type, enabled=enable_half)
epochs = 10
learning_rate = 0.001
_early_stopping = EarlyStopping(patience=10, min_delta=0.01)


def train(train_loader, optimizer, scheduler, model, epoch, **kwargs):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for batch_index, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        wandb.log({"batch_loss": loss.item(), "batch": batch_index})

    epoch_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    wandb.log({"epoch_loss": epoch_loss, "accuracy": accuracy, "epoch": epoch})
    return accuracy


def val(test_loader, model, **kwargs):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    _early_stopping(val_loss, model)

    return 100.0 * correct / total


def main():
    datasets_mapping = {
        'mnist': MNIST,
        'cifar10': CIFAR10,
        'cifar100': CIFAR100
    }

    optimizers_mapping = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop
    }

    schedulers_mapping = {
        'step_lr': optim.lr_scheduler.StepLR,
        'reduce_lr_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
        'none': None
    }

    parser = argparse.ArgumentParser(prog='TorchLine',
                                     description='Pipeline for PyTorch')

    parser.add_argument('-d', '--dataset', choices=datasets_mapping.keys(), required=True)
    parser.add_argument('-o', '--optimizer', default='sgd', choices=optimizers_mapping.keys())
    parser.add_argument('--scheduler', default='step_lr', choices=schedulers_mapping.keys())

    parser.add_argument('--sgd-momentum', type=float, default=0)
    parser.add_argument('--sgd-use-nesterov', action='store_true', default=False)
    parser.add_argument('--sgd-weight-decay', type=float, default=0)

    parser.add_argument('--augmentations', default='basic', choices=augmentations.keys())

    cifar_models = ['resnet18_cifar10', 'pre_act_resnet18']
    mnist_models = ['mlp', 'le_net']
    parser.add_argument('-m', '--model', choices=cifar_models + mnist_models, required=True)

    args = parser.parse_args()
    dataset_path = os.path.join(DATASET_PATH, args.dataset)

    if args.dataset.startswith('mnist') and args.model not in mnist_models:
        raise Exception('Dataset of type MNIST received an unexpected model')
    elif args.dataset.startswith('cifar') and args.model not in cifar_models:
        raise Exception('Dataset of type CIFAR received an unexpected model')

    match args.model:
        case 'resnet18_cifar10':
            import timm
            model = timm.create_model("hf_hub:edadaltocg/resnet18_cifar10", pretrained=False)
        case 'pre_act_resnet18':
            from models import pre_act_resnet18
            num_classes = 10 if args.dataset == 'cifar10' else 100
            model = pre_act_resnet18.PreActResNet18_C10(num_classes)
        case 'mlp':
            from torchvision.ops import MLP
            # TODO: implement and test everything
            raise NotImplementedError
        case 'le_net':
            raise NotImplementedError
        case _:
            raise Exception('Got an unexpected model name: {}'.format(args.model))

    chosen_augmentations = augmentations[args.augmentations]
    basic_transforms = v2.Compose([
        chosen_augmentations(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25), inplace=True),
    ])

    # TODO: caching component
    train_set = datasets_mapping[args.dataset](dataset_path, download=True, train=True, transform=basic_transforms)
    test_set = datasets_mapping[args.dataset](dataset_path, download=True, train=False, transform=basic_transforms)

    optimizer_class = optimizers_mapping[args.optimizer]

    if optimizer_class == optim.SGD:
        optimizer_kwargs = dict(momentum=args.sgd_momentum,
                                nesterov=args.sgd_use_nesterov,
                                weight_decay=args.sgd_weight_decay,
                                lr=learning_rate)
    else:
        optimizer_kwargs = dict(lr=learning_rate)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    match args.scheduler:
        case 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
        case 'reduce_lr_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        case 'none':
            scheduler = None
        case _:
            raise Exception('Got an unexpected scheduler name: {}'.format(args.scheduler))

    model = model.to(device)

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)

    context = {
        'train_set': train_set,
        'test_set': test_set,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
    }

    wandb.init(project="TorchLine", config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    })
    best = 0.0
    for epoch in range(epochs):
        train_acc = train(**context, epoch=epoch)
        val_acc = val(**context)
        if val_acc > best:
            best = val_acc

        print(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")

        if _early_stopping.early_stop:
            print("Triggered early stopping")
            break

    wandb.finish()


if __name__ == '__main__':
    main()
