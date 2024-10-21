import time

import torch

import utils
from config import *
from nn import NeuralNetwork

training_dataset = utils.load_dataset(train=True)
testing_dataset = utils.load_dataset(train=False)

model = NeuralNetwork()
if os.path.isfile(MODEL_STATE_DICT_PATH):
    model.load_from_disk()
all_training_instances, all_training_labels = [*utils.split_into_batches(training_dataset, 60000)][0]
all_testing_instances, all_testing_labels = [*utils.split_into_batches(testing_dataset, 10000)][0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print('Using device:', device)


def train():
    epoch = 0
    last_testing_accuracy = 0
    train_start = time.time()
    while True:
        epoch += 1
        for training_instances, training_labels in utils.split_into_batches(training_dataset, 100):
            outputs = model(training_instances)
            # print("Loss:", model.loss(training_labels))
            # print(utils.get_general_accuracy(outputs, training_labels))

            transformed_labels = torch.zeros((training_labels.shape[0], OUTPUT_DIMENSIONS_COUNT))
            transformed_labels[torch.arange(training_labels.shape[0]), training_labels] = 1

            model.backward(transformed_labels)

        # torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)

        outputs = model(all_testing_instances)
        accuracy = utils.get_general_accuracy(outputs, all_testing_labels)
        print('Testing accuracy check:', accuracy)
        print('Testing accuracy delta:', accuracy - last_testing_accuracy)
        print('Elapsed time: ', time.time() - train_start)
        print('-' * 100)
        last_testing_accuracy = accuracy
        if accuracy > 0.95:
            break


def validate():
    outputs = model(all_training_instances)
    accuracy = utils.get_general_accuracy(outputs, all_training_labels)
    f1_scores = utils.get_labels_f1_score(outputs, all_training_labels)
    print('Training general accuracy:', accuracy)
    print('Training label f1 scores:', f1_scores)
    print('-' * 100)

    outputs = model(all_testing_instances)
    accuracy = utils.get_general_accuracy(outputs, all_testing_labels, inputs=all_testing_instances)
    # accuracy = utils.get_general_accuracy(outputs, all_testing_labels)
    f1_scores = utils.get_labels_f1_score(outputs, all_testing_labels)
    print('Testing general accuracy:', accuracy)
    print('Testing label f1 scores:', f1_scores)
    print('-' * 100)


def main():
    # train()
    validate()


if __name__ == '__main__':
    main()
