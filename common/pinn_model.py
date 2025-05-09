import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    A generic PINN MLP with optional learnable PDE coefficients.

    Args:
        in_dim (int):        Dimension of input (e.g. space + time).
        hidden_dim (int):    Width of each hidden layer.
        hidden_layers (int): Number of hidden layers.
        out_dim (int):       Dimension of network output.
        pde_params (dict):   Optional dict of {name:float} to register as trainable
                             parameters (e.g. wave speed, stiffness, etc.).
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 hidden_layers: int,
                 out_dim: int,
                 pde_params: dict = None):
        super().__init__()
        # build a simple MLP: Linear–Tanh–…–Linear
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # register any PDE coefficients as learnable parameters
        # e.g. pde_params={'alpha':0.5,'c':1.0}
        self.pde_params = nn.ParameterDict()
        if pde_params is not None:
            for name, val in pde_params.items():
                self.pde_params[name] = nn.Parameter(torch.tensor(val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the MLP.
        """
        return self.net(x)

    def get_activation(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """
        Return the output (activation) of the `layer_index`-th module in `self.net`
        when feeding `x` through the network.

        Args:
            x (Tensor[N,in_dim]):      input points
            layer_index (int):         index into self.net

        Returns:
            Tensor[N,?]: the activation at that layer
        """
        activation = None
        def _hook(module, inp, out):
            nonlocal activation
            activation = out

        handle = self.net[layer_index].register_forward_hook(_hook)
        _ = self.net(x)
        handle.remove()
        return activation

    