import random
from micrograd.cleanroom_engine import Value
from typing import List

class Module:

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0

    def parameters(self) -> List[Value]:
        return []

class Neuron(Module):
    """
    `nin` values input, single output, followed by an activation function
    """

    def __init__(self, nin, nonlin=True):
        self.nin = nin
        self.nonlin = nonlin
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(0)

    def __call__(self, x):
        assert len(x) == self.nin
        # result = sum([wi * xi for (wi, xi) in zip(self.weights, x)])
        result = sum([wi * xi for (wi, xi) in zip(self.weights, x)], self.bias)
        if self.nonlin:
            result = result.relu()
        return result

    def parameters(self):
        # return self.weights
        return self.weights + [self.bias,]

    def __repr__(self):
        return f"Neuron(nin={self.nin}, nonlin={self.nonlin})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.nin = nin
        self.nout = nout
        self.nonlin = kwargs.pop("nonlin", True)
        self.neurons = [Neuron(nin, nonlin=self.nonlin) for _ in range(nout)]

    def __call__(self, x):
        assert len(x) == self.nin
        result = [neuron(x) for neuron in self.neurons]
        return result
        # return result[0] if len(result) == 1 else result

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params += neuron.parameters()
        return params

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Layer(nin={self.nin}, nout={self.nout})"

class MLP(Module):
    """
    nin is the number of input features
    nouts is the size of the hidden layers
    nouts[-1] is the number of output features
    """

    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = nouts
        self.layers = []
        sizes = [nin] + nouts
        for idx in range(1, len(sizes)):
            nonlin = True if idx != len(sizes) - 1 else False # no nonlin on final layer.
            self.layers.append(Layer(sizes[idx-1], sizes[idx], nonlin=nonlin))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x if len(x) > 1 else x[0]

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params

    def __repr__(self):
        return f"MLP(nin={self.nin}, nouts={self.nouts}, layers={self.layers})"
