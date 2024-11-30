"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            # 获取参数和梯度的 numpy 数组
            param_np = param.detach().numpy()
            grad_np = param.grad.detach().numpy()
            
            # 计算带权重衰减的梯度
            grad_np = grad_np + self.weight_decay * param_np
            
            # 初始化动量
            if i not in self.u:
                self.u[i] = np.zeros_like(grad_np)
            
            # 更新动量：u = momentum * u + grad
            self.u[i] = self.momentum * self.u[i] + grad_np
            
            # 更新参数：theta = theta - lr * u
            param.data = ndl.Tensor(param_np - self.lr * self.u[i], dtype=param.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
