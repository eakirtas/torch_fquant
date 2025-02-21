import numpy as np
import torch as T

DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class MovingAverage():
    def __call__(self, x, layer_stats, key, layer_num=None):
        if key == 'bias':
            max_val = T.max(x)
            min_val = T.min(x)
        else:
            max_val, _ = T.max(x, dim=1)
            min_val, _ = T.min(x, dim=1)

        if key not in layer_stats:
            layer_stats[key] = {'total': 1}
        else:
            layer_stats[key]['total'] += 1

        weighting = 2.0 / (
            layer_stats[key]['total']) + 1  # Exponential Moving Average

        if 'min' in layer_stats[key]:
            layer_stats[key]['min'] = weighting * (min_val.mean().item()) + (
                1 - weighting) * layer_stats[key]['min']  # EMA min
        else:
            layer_stats[key]['min'] = weighting * (min_val.mean().item())

        if 'max' in layer_stats[key]:
            layer_stats[key]['max'] = weighting * (max_val.mean().item()) + (
                1 - weighting) * layer_stats[key]['max']  # EMA max
        else:
            layer_stats[key]['max'] = weighting * (max_val.mean().item())


class StdApproach():
    def __init__(self, layers, alpha, beta=2.0):
        self.scalers = []
        self.beta = beta
        for i in range(layers):
            self.scalers.append({
                'weight':
                T.nn.Parameter(T.tensor([alpha], dtype=T.float).to(DEVICE)),
                'bias':
                T.nn.Parameter(T.tensor([alpha], dtype=T.float).to(DEVICE)),
                'input':
                T.nn.Parameter(T.tensor([alpha], dtype=T.float).to(DEVICE)),
                'act_value':
                T.nn.Parameter(T.tensor([alpha], dtype=T.float).to(DEVICE)),
            })

    def __call__(self, x, layer_stats, key, layer_num=None):
        if key == 'bias':
            average_val = T.mean(x)
            std_val = T.std(x, unbiased=False)
        else:
            average_val = T.mean(x, dim=1)
            std_val = T.std(x, dim=1, unbiased=False)

        if key not in layer_stats:

            layer_stats[key] = {'total': 1}
            weighting = self.beta / (
                layer_stats[key]['total']) + 1  # Exponential Moving Average
            layer_stats[key]['average'] = weighting * average_val.mean()
            layer_stats[key]['std'] = weighting * std_val.mean()

        else:
            layer_stats[key]['total'] += 1
            weighting = self.beta / (
                layer_stats[key]['total']) + 1  # Exponential Moving Average

            layer_stats[key]['average'] = weighting * (average_val.mean()) + (
                1 - weighting) * layer_stats[key]['average']
            layer_stats[key]['std'] = weighting * (std_val.mean()) + (
                1 - weighting) * layer_stats[key]['std']

        layer_stats[key]['min'] = layer_stats[key][
            'average'] - self.scalers[layer_num][key] * layer_stats[key]['std']

        layer_stats[key]['max'] = layer_stats[key][
            'average'] + self.scalers[layer_num][key] * layer_stats[key]['std']


class MinMax():
    def __call__(self, x, layer_stats, key, layer_num=None):
        if key == 'bias':
            max_val = T.max(x)
            min_val = T.min(x)
        else:
            max_val, _ = T.max(x, dim=1)
            min_val, _ = T.min(x, dim=1)

        if key not in layer_stats:
            layer_stats[key] = {'total': 1}
        else:
            layer_stats[key]['total'] += 1

        layer_stats[key]['min'] = (min_val.mean().item())  #Average min
        layer_stats[key]['max'] = (max_val.mean().item())  # AVERAGE max


class MinMaxStdApproach():
    def __init__(self, layers, scaler):
        self.scalers = []
        for i in range(layers):
            self.scalers.append({
                'weight':
                T.nn.Parameter(T.tensor([scaler]).to(DEVICE),
                               requires_grad=True),
                'bias':
                T.nn.Parameter(T.tensor([scaler]).to(DEVICE),
                               requires_grad=True),
                'input':
                T.nn.Parameter(T.tensor([scaler]).to(DEVICE),
                               requires_grad=True),
                'act_value':
                T.nn.Parameter(T.tensor([scaler]).to(DEVICE),
                               requires_grad=True),
            })

    def __call__(self, x, layer_stats, key, layer_num=None):
        if key == 'bias':
            average_val = T.mean(x)
            std_val = T.std(x)

        else:
            average_val = T.mean(x, dim=1)
            std_val = T.std(x, dim=1)

        if key not in layer_stats:

            layer_stats[key] = {'total': 1}
            layer_stats[key]['average'] = average_val.mean()
            layer_stats[key]['std'] = std_val.mean()

        else:
            layer_stats[key]['total'] += 1
            layer_stats[key]['average'] = (average_val.mean())
            layer_stats[key]['std'] = (std_val.mean())

        layer_stats[key]['min'] = layer_stats[key][
            'average'] - self.scalers[layer_num][key] * layer_stats[key]['std']

        layer_stats[key]['max'] = layer_stats[key][
            'average'] + self.scalers[layer_num][key] * layer_stats[key]['std']
