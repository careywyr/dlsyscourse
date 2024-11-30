from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=-1, keepdims=True)
        Z_stable = Z - Z_max
        exp_sum = array_api.sum(array_api.exp(Z_stable), axis=-1, keepdims=True)
        return Z_stable - array_api.log(exp_sum)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        log_softmax_output = node.numpy()
        softmax = array_api.exp(log_softmax_output)
        grad = out_grad - softmax * array_api.sum(out_grad.numpy(), axis=-1, keepdims=True)
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_stable = Z - Z_max
        exp_sum = array_api.sum(array_api.exp(Z_stable), axis=self.axes, keepdims=True)
        log_sum_exp = array_api.log(exp_sum) + Z_max

        if self.axes is not None:
            new_shape = [dim for i, dim in enumerate(Z.shape) if i not in self.axes]
            log_sum_exp = log_sum_exp.reshape(new_shape)
        else:
            log_sum_exp = float(log_sum_exp)

        return log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        Z_max = array_api.max(Z.numpy(), axis=self.axes, keepdims=True)
        Z_stable = Z.numpy() - Z_max
        exp_Z_stable = array_api.exp(Z_stable)
        exp_sum = array_api.sum(exp_Z_stable, axis=self.axes, keepdims=True)

        softmax_grad = exp_Z_stable / exp_sum

        if self.axes is not None:
            new_shape = [1 if i in self.axes else dim for i, dim in enumerate(Z.shape)]
            out_grad = out_grad.reshape(new_shape)

        return out_grad * softmax_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
