import random

from torch.utils.data.dataset import T_co
import itertools
import functools

from PIL import Image
from torchvision.transforms import ToTensor

from config import *
import torch.utils.data

from torchvision.transforms import v2
import torchvision.transforms.functional


@functools.cache
def load_image(image_path: str, transforms: v2.Transform = None):
    image = Image.open(image_path)
    image = ToTensor()(image)
    return image if transforms is None else transforms(image)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir=DATASET_DIR, transforms: v2.Transform = None, use_random_rotation=True, dataset_cache=None, indices_cache=None, *args, **kwargs):
        super(MyDataset, self).__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        # TODO: augmentations

        self.dataset_size = None
        self.use_random_rotation = use_random_rotation

        if dataset_cache and indices_cache:
            self.dataset = dataset_cache
            self.indices = indices_cache
        else:
            self.dataset = []
            self.indices = []
            self.load_dataset()

    def load_dataset(self):
        global_index = 0
        for series in os.listdir(self.dataset_dir):  # L15-0331E-1257N_1327_3160_13
            series_dir = os.path.join(self.dataset_dir, series, 'images')

            files = sorted(os.listdir(series_dir), key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3])))

            for i, image_name in enumerate(files):
                image_path = os.path.join(series_dir, image_name)

                parts = image_name.split('_')
                year = int(parts[2])
                month = int(parts[3])

                normalized_month = year * 12 + month

                self.dataset.append((image_path, normalized_month))

            start_index = global_index
            end_index = start_index + len(files)

            for i, j in itertools.combinations([*range(start_index, end_index)], 2):
                self.indices.append((i, j))

            global_index = end_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index) -> T_co:
        if isinstance(index, slice):
            indices = [self.indices[i] for i in range(index.start or 0, index.stop or len(self), index.step or 1)]
            dataset_copy = MyDataset(self.dataset_dir, self.transforms, self.use_random_rotation, dataset_cache=self.dataset, indices_cache=indices)
            return dataset_copy

        if index < 0 or index >= len(self):
            raise IndexError

        i, j = self.indices[index]
        image_path_1, months_1 = self.dataset[i]
        image_path_2, months_2 = self.dataset[j]

        image_1 = load_image(image_path_1, self.transforms)
        image_2 = load_image(image_path_2, self.transforms)

        if self.use_random_rotation:
            angle = (random.random() * 60) - 30
            image_1 = torchvision.transforms.functional.rotate(image_1, angle, center=[image_1.shape[1] / 2, image_1.shape[2] / 2])
            image_2 = torchvision.transforms.functional.rotate(image_2, angle, center=[image_2.shape[1] / 2, image_2.shape[2] / 2])

        return image_1, image_2, months_2 - months_1
