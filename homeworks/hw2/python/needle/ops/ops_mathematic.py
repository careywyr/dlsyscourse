"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy
import needle as ndl

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad * (rhs * array_api.power(lhs, rhs - 1)) 

        # (lhs ** rhs) * log(lhs)
        grad_b = out_grad * array_api.power(lhs, rhs) * array_api.log(lhs.numpy())
        return lhs_grad, ndl.Tensor(grad_b)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return (out_grad * self.scalar * power_scalar(lhs, self.scalar - 1), )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = array_api.divide(out_grad, rhs)
        rhs_grad = -array_api.divide(out_grad * lhs, array_api.power(rhs, 2))
        return (lhs_grad, rhs_grad)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar, )
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 如果 axes 是 None，默认交换最后两个轴
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return array_api.swapaxes(a, x, y)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            # 如果 axes 是 None，默认交换最后两个轴
            x, y = -1, -2
        return transpose(out_grad, axes=(x, y))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 将梯度 reshape 成原来的形状
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs_shape = node.inputs[0].shape  # 原始输入的形状
        output_shape = self.shape  # 广播后的目标形状

        # 如果输入形状和输出形状相同，说明没有发生广播，直接返回 out_grad
        if lhs_shape == output_shape:
            return out_grad

        # 计算输入和输出形状的维度差，处理输入比输出少的维度情况
        ndim_diff = len(output_shape) - len(lhs_shape)
        
        # 需要缩减的维度
        reduce_axes = []

        # 遍历输出张量的每个维度，找到需要缩减的维度
        for i in range(len(output_shape)):
            # 计算对应的输入维度，如果是补上去的（即左边加 1），则视为 1
            input_dim = lhs_shape[i - ndim_diff] if i >= ndim_diff else 1
            output_dim = output_shape[i]

            # 如果输入维度为 1，而输出维度大于 1，则需要在该轴进行求和
            if input_dim == 1 and output_dim != 1:
                reduce_axes.append(i)

        # 如果 reduce_axes 非空，执行 sum 操作以消除广播的影响
        if reduce_axes:
            # out_grad = array_api.sum(out_grad.numpy(), axis=tuple(reduce_axes))
            out_grad = summation(out_grad, axes=tuple(reduce_axes))

        # reshape 回原始输入的形状
        return reshape(out_grad, lhs_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]  
        input_shape = lhs.shape  # 原始张量的形状

        if out_grad.shape == ():
            out_grad = array_api.full_like(lhs.numpy(), out_grad.numpy())
        
        if self.axes is None:
            # 如果没有指定轴，意味着我们对整个张量求和，结果是一个标量
            return broadcast_to(ndl.Tensor(out_grad), input_shape)

        out_grad = out_grad.numpy() if isinstance(out_grad, Tensor) else out_grad

        # 为 out_grad 恢复被求和的维度
        if out_grad.shape != input_shape:
          for axis in sorted(self.axes, reverse=True):
              out_grad = array_api.expand_dims(out_grad, axis)

        # 扩展为原始输入的形状
        result = broadcast_to(ndl.Tensor(out_grad), input_shape)
        return result
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs

        # 始终使用批量矩阵乘法的逻辑
        grad_lhs = matmul(out_grad, transpose(rhs, axes=(-2, -1)))
        grad_rhs = matmul(transpose(lhs, axes=(-2, -1)), out_grad)

        # 处理广播的维度
        if len(lhs.shape) < len(rhs.shape):
            axes_to_sum = tuple(range(len(rhs.shape) - len(lhs.shape)))
            grad_lhs = summation(grad_lhs, axes=axes_to_sum)
        if len(rhs.shape) < len(lhs.shape):
            axes_to_sum = tuple(range(len(lhs.shape) - len(rhs.shape)))
            grad_rhs = summation(grad_rhs, axes=axes_to_sum)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad, )
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad / lhs
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return out_grad * exp(lhs)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 获取前向传播的输入
        lhs = node.inputs[0]
        np_lhs = lhs.numpy()
        # 创建一个掩码：lhs > 0 的位置为 1，其他位置为 0
        grad_mask = array_api.where(np_lhs > 0, 1, 0)
        # 将 out_grad 和 grad_mask 相乘，得到最终的梯度
        return out_grad * ndl.Tensor(grad_mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

