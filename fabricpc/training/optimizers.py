from abc import ABC, abstractmethod
import torch


def get_optimizer_class(type: str):
    type = type.lower()
    if type == "adam":
        return Adam
    elif type == "sgd":
        return SGD
    else:
        raise ValueError(f"Unknown optimizer type '{type}'. Supported: 'adam', 'sgd'.")


class BaseOptim(ABC):
    @abstractmethod
    def __init__(self, param: torch.Tensor, **kwargs):
        pass

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def step(self, grad: torch.Tensor):
        pass

    @abstractmethod
    def to(self, device: torch.device | str):
        pass


class Adam(BaseOptim):
    """
    Minimal Adam optimizer to be attached to a parameter tensor.
    Usage:
        opt = Adam(edge.W, lr=1e-3)
        ...
        opt.step(grad)  # where grad is the gradient w.r.t. edge.W
    """

    def __init__(
        self,
        param: torch.Tensor,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        if not isinstance(param, torch.Tensor):
            raise TypeError("param must be a torch.Tensor")
        if not param.is_floating_point():
            raise TypeError("param must be a floating point tensor")
        self.param = param
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        # State
        self.t = 0
        self.m = torch.zeros_like(self.param, device=self.param.device)
        self.v = torch.zeros_like(self.param, device=self.param.device)

    def reset_state(self):
        """Reset moment estimates and time step."""
        self.t = 0
        self.m.zero_()
        self.v.zero_()

    @torch.no_grad()
    def step(self, grad: torch.Tensor):
        """
        Perform a single Adam update step using the provided gradient.

        Args:
            grad: gradient tensor with the same shape as self.param
        """
        if grad is None:
            return
        if not isinstance(grad, torch.Tensor):
            raise TypeError("grad must be a torch.Tensor")
        if grad.shape != self.param.shape:
            raise ValueError(
                f"grad shape {grad.shape} does not match param shape {self.param.shape}"
            )
        if grad.dtype != self.param.dtype:
            grad = grad.to(self.param.dtype)
        if grad.device != self.param.device:
            grad = grad.to(self.param.device)

        # Optional L2 regularization added to gradient
        if self.weight_decay != 0.0:
            grad = grad + self.weight_decay * self.param

        self.t += 1

        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)

        # Bias correction
        bias_correction1 = 1.0 - (self.beta1**self.t)
        bias_correction2 = 1.0 - (self.beta2**self.t)

        m_hat = self.m / bias_correction1
        v_hat = self.v / bias_correction2

        # Parameter update with normal operators
        denom = torch.sqrt(v_hat) + self.eps
        step = m_hat / denom
        self.param -= self.lr * step

        # print(f"Otpim step Adam lr = {self.lr} param dims {self.param.shape}")

    def to(self, device: torch.device | str):
        """Move optimizer state to the specified device (param must already be on that device)."""
        device = torch.device(device)
        if self.param.device != device:
            raise ValueError(
                "Move the parameter to the target device before moving optimizer state."
            )
        self.m = self.m.to(device)
        self.v = self.v.to(device)


# ... existing code ...
class SGD(BaseOptim):
    """
    Stochastic Gradient Descent optimizer with optional momentum, dampening, Nesterov, and weight decay.
    Usage:
        opt = SGD(edge.W, lr=1e-2, momentum=0.9)
        ...
        opt.step(grad)  # where grad is the gradient w.r.t. edge.W
    """

    def __init__(
        self,
        param: torch.Tensor,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        if not isinstance(param, torch.Tensor):
            raise TypeError("param must be a torch.Tensor")
        if not param.is_floating_point():
            raise TypeError("param must be a floating point tensor")
        if nesterov and (momentum <= 0 or dampening != 0.0):
            raise ValueError(
                "Nesterov momentum requires momentum > 0 and dampening = 0"
            )

        self.param = param
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.dampening = float(dampening)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)

        # State
        self.buf = None  # momentum buffer

    def reset_state(self):
        """Reset optimizer state."""
        self.buf = None

    def _get_buf(self):
        if self.buf is None:
            self.buf = torch.zeros_like(self.param)
        return self.buf

    @torch.no_grad()
    def step(self, grad: torch.Tensor):
        """
        Perform a single SGD update step using the provided gradient.

        Args:
            grad: gradient tensor with the same shape as self.param
        """
        if grad is None:
            return
        if not isinstance(grad, torch.Tensor):
            raise TypeError("grad must be a torch.Tensor")
        if grad.shape != self.param.shape:
            raise ValueError(
                f"grad shape {grad.shape} does not match param shape {self.param.shape}"
            )
        if grad.dtype != self.param.dtype:
            grad = grad.to(self.param.dtype)
        if grad.device != self.param.device:
            grad = grad.to(self.param.device)

        d_p = grad

        # L2 regularization
        if self.weight_decay != 0.0:
            d_p += self.weight_decay * self.param

        if self.momentum != 0.0:
            buf = self._get_buf()
            # v_t = momentum * v_{t-1} + (1 - dampening) * grad
            buf = self.momentum * buf + (1.0 - self.dampening) * d_p
            self.buf = buf  # persist buffer
            if self.nesterov:
                # d_p = grad + momentum * v_t
                d_p += self.momentum * buf
            else:
                d_p = buf

        # Param update with normal operators
        self.param -= self.lr * d_p

    def to(self, device: torch.device | str):
        """Move optimizer state to the specified device (param must already be on that device)."""
        device = torch.device(device)
        if self.param.device != device:
            raise ValueError(
                "Move the parameter to the target device before moving optimizer state."
            )
        if self.buf is not None:
            self.buf = self.buf.to(device)


def instantiate_optimizer(W: torch.Tensor, optim_cfg: dict):
    """
    Instantiate optimizer for tensor from a config:
        W = torch.Tensor(), weights to optimize
        optimizer_cfg = {'type': 'adam'|'sgd', 'config': {<kwargs>}}

      - instantiate_optimizer(W, {'type': 'adam', 'lr': 1e-2})
      - instantiate_optimizer(W, {'type': 'sgd', 'lr': 1e-2, 'momentum': 0.9})
      - instantiate_optimizer(W)  # defaults to Adam(lr=default_lr)
    Falls back to Adam with defaults if not provided.

    Returns:
      An optimizer instance bound to 'W'.
    """
    # Local mapping to avoid stringly-typed imports in user configs
    if "type" not in optim_cfg:
        raise ValueError("config['type'] is required")
    type = optim_cfg["type"].lower()
    config = {k: v for k, v in optim_cfg.items() if k != "type"}

    # Factory
    OptimCls = get_optimizer_class(type)

    # Instantiate one optimizer per edge (using its weight tensor)
    return OptimCls(W, **config)
