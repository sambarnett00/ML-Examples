import numpy as np
from typing import List, Tuple


class Network:
    """Interface class, necessity TBD"""
    def __init__(self, layout, weights=None) -> None:
        ...
    def forward(self, x):
        ...
    def get_weights(self):
        ...
    def set_weights(self, weights) -> None:
        ...

# Activation functions
def _sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

def _relu(z): 
    return np.maximum(0, z)

def _softmax(z):
    exp = np.exp(z)
    return exp / exp.sum()

def _tanh(z):
    return np.tanh(z)


class NumpyNetwork(Network):
    ACT_MAP = {"sigmoid": _sigmoid, "relu": _relu, "softmax": _softmax, "tanh": _tanh}
    
    def __init__(self, 
        layout: List[Tuple[int, str]], ## [(10, None), (32, 'relu'), (64, 'relu'), ...]
        weights: List[np.ndarray] = None) -> None:
        
        self.layout = layout
        if weights is None:
            self.weights = [np.random.randn(l1, l2) for (l1, _), (l2, _) in zip(layout[:-1], layout[1:])]
        else:
            self.weights = weights
            
        self.act_funcs = [NumpyNetwork.ACT_MAP[act.strip().lower()] for _, act in layout[1:]] ## input layer has no activation


    def forward(self, x: np.ndarray) -> np.ndarray:
        for W, func in zip(self.weights, self.act_funcs):
            x = func(x @ W)
            
        return x

    def set_weights(self, layout: List[Tuple[int, str]], weights: List[np.ndarray]) -> None:
        self.layout = layout
        self.weights = weights
        self.act_funcs = [NumpyNetwork.ACT_MAP[act.strip().lower()] for _, act in layout[1:]]

    def get_weights(self) -> List[np.ndarray]:
        return self.weights


class TorchNetwork(Network):
    """TODO"""
    ...
