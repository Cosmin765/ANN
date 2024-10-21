import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from config import *


def load_dataset(train=True):
    def image_to_tensor(image):
        np_image = np.array(image, dtype=np.float32)
        tensor = np.array(np_image, dtype=np.float32) / 255.0
        return tensor

    dataset_path = TRAINING_DATASET_PATH if train else TESTING_DATASET_PATH
    dataset = torchvision.datasets.MNIST(root=dataset_path, download=True, train=train)
    dataset.data = torch.tensor([image_to_tensor(image) for image in dataset.data])
    return dataset


def split_into_batches(dataset: torchvision.datasets.MNIST, batch_size=1000):
    instances = dataset.data
    labels = dataset.targets

    indices = torch.randperm(instances.shape[0])
    instances, labels = instances[indices], labels[indices]  # shuffle the data

    for i in range(0, instances.shape[0], batch_size):
        # reshape instances
        instances_batch = torch.tensor(np.array([instance.reshape(INSTANCE_DIMENSIONS_COUNT) for instance in instances[i:i+batch_size]]))
        labels_batch = torch.tensor(np.array(labels[i:i+batch_size]))
        yield instances_batch, labels_batch


def get_general_accuracy(outputs, labels, inputs=None):
    matched = 0
    for index, (output, label) in enumerate(zip(outputs, labels)):
        prediction = np.argmax(output.detach().numpy())

        if inputs is not None:
            # for my ego
            print(prediction)
            display_image(inputs[index])

        if prediction == label:
            matched += 1

    return matched / len(labels)


def get_labels_f1_score(outputs, labels):
    """
        Returns the F1 scores for each label, as a list
    """

    tps = [0] * OUTPUT_DIMENSIONS_COUNT
    tns = [0] * OUTPUT_DIMENSIONS_COUNT
    fps = [0] * OUTPUT_DIMENSIONS_COUNT
    fns = [0] * OUTPUT_DIMENSIONS_COUNT

    for output, label in zip(outputs, labels):
        prediction = np.argmax(output.detach().numpy())
        if prediction == label:
            for target in range(OUTPUT_DIMENSIONS_COUNT):
                if prediction == target:
                    tps[target] += 1
                else:
                    tns[target] += 1
        else:
            fps[prediction] += 1
            fns[label] += 1

            for target in range(OUTPUT_DIMENSIONS_COUNT):
                if target in {prediction, label}:
                    continue

                tns[target] += 1

    f1_scores = []

    for tp, tn, fp, fn in zip(tps, tns, fps, fns):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / ((precision + recall) or 0.01)
        f1_scores.append(f1_score)

    return f1_scores


def display_image(data):
    plt.imshow(data.reshape(28, 28))
    plt.show()


def mean_squared_error(output: torch.Tensor, labels: torch.Tensor):
    n = output.shape[0]
    return (1 / (2 * n)) * np.sum((output - labels) ** 2)
