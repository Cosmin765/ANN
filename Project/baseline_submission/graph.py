import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


with open('metrics_baseline.json') as fd:
    metrics_baseline = json.load(fd)

with open('metrics_improved.json') as fd:
    metrics_improved = json.load(fd)


def convert_memory(*values_all):
    return [[v / (1 << 30) for v in values] for values in values_all]


for key in metrics_baseline:
    values_baseline = metrics_baseline[key]
    values_improved = metrics_improved[key]

    integer = False
    unit = '(%)'
    if key == 'memory':
        values_baseline, values_improved = convert_memory(values_baseline, values_improved)
        unit = '(GB)'
    elif 'loss' in key:
        unit = ''
    elif key == 'timestamp':
        unit = ''

    plt.figure(figsize=(12, 8))

    plt.plot(values_baseline, label=key + "_baseline")
    plt.plot(values_improved, label=key + "_improved")

    plt.title(key.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel(f'Value {unit}')
    plt.legend()
    plt.grid(True)

    # plt.show()
    plt.savefig(key + '.png')
