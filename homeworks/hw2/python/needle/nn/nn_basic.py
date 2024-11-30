"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype)
        self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1), (1, out_features)) , device=device, dtype=dtype) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)
        if self.bias:
            self.bias = ops.broadcast_to(self.bias, output.shape)
            output += self.bias
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # 获取批次大小
        batch_size = X.shape[0]
        # 计算展平后的维度大小：所有非批次维度的乘积
        flattened_size = 1
        for i in range(1, len(X.shape)):
            flattened_size *= X.shape[i]
        # 重塑为 (batch_size, flattened_size)
        return ops.reshape(X, (batch_size, flattened_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module.forward(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[0]  # batch size
        k = logits.shape[1]  # number of classes
        y_one_hot = init.one_hot(k, y)  # convert labels to one-hot
        
        # 计算 logsumexp
        logsumexp = ops.logsumexp(logits, axes=(1,))  # shape: (n,)
        
        # 选择对应标签的 logits
        z_y = ops.summation(logits * y_one_hot, axes=(1,))  # shape: (n,)
        
        # 计算损失
        loss = ops.summation(logsumexp - z_y) / n
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # 初始化可学习参数
        self.weight = Parameter(init.ones(dim, requires_grad=True), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim, requires_grad=True), device=device, dtype=dtype)
        
        # 初始化运行统计量
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        
        if self.training:
            # 计算批次统计量
            mean = ops.summation(x, axes=(0,)) / batch_size  # shape: (dim,)
            
            # 计算方差，使用有偏估计（除以 N）
            x_centered = x - ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
            var = ops.summation(x_centered ** 2, axes=(0,)) / batch_size  # shape: (dim,)
            
            # 更新运行统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            # 使用运行统计量
            mean = self.running_mean
            var = self.running_var
            x_centered = x - ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)
        
        # 归一化
        norm = x_centered / ops.broadcast_to(ops.reshape(ops.power_scalar(var + self.eps, 0.5), (1, self.dim)), x.shape)
        
        # 应用可学习参数
        return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * norm + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # 初始化权重为 1
        self.weight = Parameter(init.ones(dim, requires_grad=True), device=device, dtype=dtype)
        # 初始化偏置为 0
        self.bias = Parameter(init.zeros(dim, requires_grad=True), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 计算均值
        mean = ops.summation(x, axes=(1,)) / self.dim  # shape: (batch_size,)
        
        # 计算方差，使用有偏估计（除以 N）
        x_centered = x - ops.broadcast_to(ops.reshape(mean, (mean.shape[0], 1)), x.shape)
        var = ops.summation(x_centered ** 2, axes=(1,)) / self.dim  # shape: (batch_size,)
        
        # 归一化
        norm = x_centered / ops.broadcast_to(ops.reshape(ops.power_scalar(var + self.eps, 0.5), (var.shape[0], 1)), x.shape)
        
        # 应用可学习参数
        return ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) * norm + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # 在训练模式下：
            # 1. 生成与输入相同形状的随机掩码（伯努利分布）
            # 2. 将选中的元素置零，并对其他元素进行缩放
            mask = init.randb(*x.shape, p=1-self.p)  # 1-p 的概率生成 1，p 的概率生成 0
            return x * mask / (1 - self.p)  # 缩放以保持期望值不变
        else:
            # 在测试模式下直接返回输入
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
