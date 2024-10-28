from my_dataset import MyDataset
import torch
from torchvision.transforms import v2
import torch.utils.data
from my_model import ImageTransformationPredictor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)


transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ColorJitter(hue=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = MyDataset(transforms=transforms, use_random_rotation=False)

dataset_size = len(dataset)

# pentru test, incercam cu un batch size mai mic
# train_dataset = dataset[:int(dataset_size * 0.7)]
train_dataset = dataset[:int(dataset_size * 0.01)]
test_dataset = dataset[int(dataset_size * 0.7):int(dataset_size * 0.85)]
validation_dataset = dataset[int(dataset_size * 0.85):]

print('train', len(train_dataset))
print('test', len(test_dataset))
print('validation', len(validation_dataset))

model = ImageTransformationPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


def train():
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, generator=torch.Generator(device='cuda')) if device == 'cuda' \
        else torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        image1, image2, diff_months = batch
        image1, image2, diff_months = image1.to(device), image2.to(device), diff_months.to(device).to(torch.float)

        optimizer.zero_grad()

        outputs = model(image1, image2, diff_months)

        loss = loss_fn(outputs, image2)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image1.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss


def validate():
    dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True, generator=torch.Generator(device='cuda')) if device == 'cuda' \
        else torch.utils.data.DataLoader(validation_dataset, batch_size=512, shuffle=True)
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            image1, image2, diff_months = batch
            image1, image2, diff_months = image1.to(device), image2.to(device), diff_months.to(device).to(torch.float)

            outputs = model(image1, image2, diff_months)

            loss = loss_fn(outputs, image2)

            running_loss += loss.item() * image1.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Validation Loss: {epoch_loss:.4f}")
    return epoch_loss


def run(epochs):
    for _ in range(epochs):
        train()
        validate()


def main():
    run(100)


if __name__ == '__main__':
    main()
