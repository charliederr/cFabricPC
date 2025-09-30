from abc import ABC, abstractmethod
import torch


def get_activation(config: dict):
    """
    Instantiate an activation function and its derivative from a config dictionary.
    The config dictionary must contain a 'type' key specifying the activation function name,
    and may contain additional keys for parameters required by certain activation functions.
    Supported activation functions:
        - 'identity': f(x) = x
        - 'sigmoid': f(x) = 1 / (1 + exp(-x))
        - 'tanh': f(x) = tanh(x)
        - 'relu': f(x) = max(0, x)
        - 'leaky_relu': f(x) = max(alpha * x, x), requires 'alpha' parameter (default 0.01)
        - 'hard_tanh': f(x) = clip(x, min_val, max_val), requires 'min_val' and 'max_val' parameters (default -1 and 1)
    Returns:
        - f: callable activation function
        - f_derivative: callable derivative function
    Example usage:
        f, f_derivative = get_activation({'type': 'leaky_relu', 'alpha': 0.02})
    """
    if "type" not in config:
        raise ValueError("config['type'] is required")
    type = config["type"].lower()
    config = {
        k: v for k, v in config.items() if k != "type"
    }  # skip the 'type' key in arguments

    if type == "identity":
        f = Identity()
    elif type == "sigmoid":
        f = Sigmoid()
    elif type == "tanh":
        f = Tanh()
    elif type == "relu":
        f = Relu()
    elif type == "leaky_relu":
        f = Leaky_relu(**config)
    elif type == "hard_tanh":
        f = Hard_tanh(**config)
    else:
        raise ValueError(
            f"Unknown activation function '{type}'. Supported: 'identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'hard_tanh'."
        )
    # Each activation function class has two methods: __call__ and derivative
    return f, f.derivative


class BaseAct(ABC):
    def __init__(self):
        super().__init__()
        self.type = ""  # the class type to instantiate

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass


class Identity(BaseAct):
    def __call__(self, x):
        return x

    def derivative(self, x):
        return torch.ones_like(x)


class Sigmoid(BaseAct):
    def __call__(self, x):
        return torch.sigmoid(x)  # 1 / (1 + torch.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))


class Tanh(BaseAct):
    def __call__(self, x):
        return torch.tanh(x)

    def derivative(self, x):
        return 1 - self(x) ** 2


class Relu(BaseAct):
    def __call__(self, x):
        return torch.relu(x)

    def derivative(self, x):
        return (x > 0).float()  # slope is unity for values above zero


class Leaky_relu(BaseAct):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return torch.max(self.alpha * x, x)

    def derivative(self, x):
        return (x > 0).float() + self.alpha * (
            x < 0
        ).float()  # slope is the leakrate r values below below zero and unity for values above zero


class Hard_tanh(BaseAct):
    def __init__(self, min_val=-1, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        # Use clamp to avoid tensor–scalar torch.min/torch.max type issues
        return torch.clamp(
            x, min=self.min_val, max=self.max_val
        )  # return x if between bounds, else clip

    def derivative(self, x):
        return (x > self.min_val) * (x < self.max_val)  # slope is unity between bounds
