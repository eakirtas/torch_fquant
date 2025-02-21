from collections import namedtuple

import torch as T

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def calc_zero_point(min_val, max_val, num_bits):
    q_min = 0.
    q_max = 2.**num_bits - 1.

    scale = (max_val - min_val) / (q_max - q_min)

    if scale == 0:
        initial_zero_point = q_min
    else:
        initial_zero_point = q_min - min_val / scale

    zero_point = 0

    if scale == 0:
        zero_point = 0
    elif initial_zero_point < q_min:
        zero_point = q_min
    elif initial_zero_point > q_max:
        zero_point = q_max
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point, q_min, q_max


def quantize_tensor(x, num_bits, min_val=None, max_val=None):
    # print('==================== EXPECTED START ====================')
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    scale, zero_point, q_min, q_max = calc_zero_point(min_val, max_val,
                                                      num_bits)

    # print("Scale: {}, Zero point: {}, q_min:{}, q_max:{}, v_min:{}, v_max:{}".
    #       format(scale, zero_point, q_min, q_max, min_val, max_val))

    q_x = zero_point + x / scale
    q_x.clamp_(q_min, q_max).round_()
    q_x = q_x.round().byte()

    # print('==================== EXPECTED ENDS ====================')
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


class FakeQuant1(T.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits, min_val=None, max_val=None):
        x = quantize_tensor(x,
                            num_bits=num_bits,
                            min_val=min_val,
                            max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
