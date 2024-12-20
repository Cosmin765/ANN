import torch

from config import *


class NeuralNetwork:
    def __init__(self, hidden_neurons=100, learning_rate=LEARNING_RATE):
        self.input_neurons = INSTANCE_DIMENSIONS_COUNT
        self.hidden_neurons = hidden_neurons
        self.output_neurons = OUTPUT_DIMENSIONS_COUNT

        self.layer_1 = torch.randn((self.input_neurons, self.hidden_neurons)) * 0.01
        self.bias_hidden = torch.zeros((1, self.hidden_neurons), dtype=torch.float32)
        self.layer_2 = torch.randn((self.hidden_neurons, self.output_neurons)) * 0.01
        self.bias_output = torch.zeros((1, self.output_neurons), dtype=torch.float32)

        self.hidden_result = None
        self.output_result = None

        self.learning_rate = learning_rate
        self.instances = None

    @staticmethod
    def relu(x: torch.Tensor):
        return torch.maximum(x, torch.tensor(0.0))

    @staticmethod
    def softmax(x: torch.Tensor):
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        return exp_x / exp_x.sum(dim=1, keepdim=True)

    def forward(self, instances: torch.Tensor):
        self.instances = instances
        x = instances @ self.layer_1 + self.bias_hidden
        x = self.relu(x)
        self.hidden_result = x
        x = x @ self.layer_2 + self.bias_output
        self.output_result = self.softmax(x)
        return self.output_result

    def loss(self, labels: torch.Tensor):
        return torch.nn.functional.cross_entropy(self.output_result, labels)

    def backward(self, labels: torch.Tensor):
        deltas = self.output_result - labels
        batch_size = labels.size(0)

        self.layer_2 -= self.learning_rate * (self.hidden_result.T @ deltas) / batch_size
        self.bias_output -= self.learning_rate * torch.mean(deltas, dim=0, keepdim=True)

        derivative = (self.hidden_result > 0).float()  # for ReLU
        deltas_new = (deltas @ self.layer_2.T) * derivative

        self.layer_1 -= self.learning_rate * (self.instances.T @ deltas_new) / batch_size
        self.bias_hidden -= self.learning_rate * torch.mean(deltas_new, dim=0, keepdim=True)

    def load_from_disk(self):
        state_dict = torch.load(MODEL_STATE_DICT_PATH, weights_only=True)
        self.layer_1 = state_dict['layer_1']
        self.layer_2 = state_dict['layer_2']
        self.bias_hidden = state_dict['bias_hidden']
        self.bias_output = state_dict['bias_output']

    def __call__(self, instances: torch.Tensor):
        return self.forward(instances)

    def state_dict(self):
        return {
            'layer_1': self.layer_1,
            'layer_2': self.layer_2,
            'bias_hidden': self.bias_hidden,
            'bias_output': self.bias_output,
        }
