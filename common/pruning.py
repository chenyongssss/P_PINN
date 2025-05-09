import numpy as np
import torch
import torch.nn as nn

def prune_neuron(model: nn.Module,
                 layer_index: int,
                 neuron_idx: int):
    """
    Zero out the weights & bias of neuron `neuron_idx` in layer `layer_index`,
    and also zero its incoming connections in the next linear layer.
    """
    # find the Linear layer to prune
    linear = model.net[layer_index]
    if not isinstance(linear, nn.Linear):
        print(f"Layer {layer_index} is not Linear; skipping")
        return

    # locate the next Linear layer for incoming connections
    next_lin = None
    for m in model.net[layer_index+1:]:
        if isinstance(m, nn.Linear):
            next_lin = m
            break

    with torch.no_grad():
        # zero out outgoing weights & bias
        linear.weight[neuron_idx, :].zero_()
        if linear.bias is not None:
            linear.bias[neuron_idx].zero_()
        # zero out incoming weights in the next layer
        if next_lin is not None:
            next_lin.weight[:, neuron_idx].zero_()

def selective_pruning_layer(model: nn.Module,
                            layer_index: int,
                            retain_data: tuple,
                            forget_data: tuple,
                            alpha: float = 0.01,
                            num_iter: int = 3):
    """
    Perform selective pruning on a single layer:

      For `num_iter` rounds:
        1. compute activations on X_good and X_bad via model.get_activation.
        2. score each neuron by (mean_bad - mean_good) normalized by overall mean
        3. prune top-α fraction of highest-scoring neurons via prune_neuron

    Args:
        model:         PINN instance
        layer_index:   index into model.net of the Linear to prune
        retain_data:   (X_good, u_good) tuple, not used here except for shape
        forget_data:   (X_bad,  u_bad)
        alpha:         fraction of neurons to remove each iteration
        num_iter:      number of pruning passes
    Returns:
        model (in-place pruned)
    """
    X_good, _ = retain_data
    X_bad, _ = forget_data

    for it in range(num_iter):
        # extract activations
        acts_g = model.get_activation(X_good, layer_index).detach().cpu().numpy()
        acts_b = model.get_activation(X_bad,  layer_index).detach().cpu().numpy()

        num_neurons = acts_g.shape[1]
        scores = np.zeros(num_neurons)
        # build score = (mean_bad - mean_good)
        for n in range(num_neurons):
            mg = acts_g[:, n].mean()
            mb = acts_b[:, n].mean()
            scores[n] = mb - mg

        # normalize by overall mean magnitude
        c = np.mean(np.abs(np.vstack([acts_g, acts_b]))) + 1e-8
        scores /= c

        k = max(int(alpha * num_neurons), 1)
        prune_idx = np.argsort(scores)[-k:]
        print(f"[Prune] layer={layer_index} iter={it} prune_idxs={prune_idx}, scores={scores[prune_idx]}")
        for idx in prune_idx:
            prune_neuron(model, layer_index, int(idx))

    return model

def selective_pruning_multi_layers(model: nn.Module,
                                   layer_indices: list[int],
                                   retain_data: tuple,
                                   forget_data: tuple,
                                   alpha: float = 0.01,
                                   num_iter: int = 3):
    """
    Apply `selective_pruning_layer` in sequence to each index in `layer_indices`.
    """
    for li in layer_indices:
        model = selective_pruning_layer(model, li, retain_data, forget_data, alpha, num_iter)
    return model
