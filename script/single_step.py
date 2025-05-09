import numpy as np
import torch
import torch.nn as nn


def prune_neuron(model, layer_index, neuron_idx):
    """
    Zero out the weights corresponding to the specified neuron in the given layer.
    """
    linear_layer = model.net[layer_index]
    next_layer = None
    for i in range(layer_index + 1, len(model.net)):
        if isinstance(model.net[i], nn.Linear):
            next_layer = model.net[i]
            break
    if next_layer is None:
        print("No subsequent linear layer found; skipping pruning")
        return
    with torch.no_grad():
        linear_layer.weight[neuron_idx, :] = 0.0
        if linear_layer.bias is not None:
            linear_layer.bias[neuron_idx] = 0.0
        next_layer.weight[:, neuron_idx] = 0.0
def selective_pruning_layer_single_step(model, layer_index, retain_data, forget_data, alpha=0.01, epsilon=1e-8):
    """
    Single-step pruning strategy: Calculate scores and prune neurons for a specified layer in one go.
    Score calculation is the same as in the iterative version:
      score(n) = (mean(activations in forget) - mean(activations in retain)) / (mean(|activations|) + epsilon)
    Then select the top k = alpha * (total number of neurons) neurons with the highest scores for pruning.
    """
    X_good = retain_data[0]
    X_bad = forget_data[0]
    acts_good = model.get_activation(X_good, layer_index).detach().cpu().numpy()
    acts_bad = model.get_activation(X_bad, layer_index).detach().cpu().numpy()
    num_neurons = acts_good.shape[1]
    scores = np.zeros(num_neurons)
    for n in range(num_neurons):
        mg = np.mean(acts_good[:, n])
        mb = np.mean(acts_bad[:, n])
        scores[n] = mb - mg
    c = np.mean(np.abs(np.concatenate([acts_good, acts_bad], axis=0))) + epsilon
    scores = scores / c
    k = int(alpha * num_neurons)
    prune_idx = np.argsort(scores)[-k:]
    print(f"Layer {layer_index} single-step prune: Prune neuron indices = {prune_idx}, scores = {scores[prune_idx]}")
    for idx in prune_idx:
        prune_neuron(model, layer_index, idx)
    return model

def selective_pruning_multi_layers_single_step(model, layer_indices, retain_data, forget_data, alpha=0.01, epsilon=1e-8):
    """
    Apply single-step pruning to multiple specified layers.
    """
    for li in layer_indices:
        model = selective_pruning_layer_single_step(model, li, retain_data, forget_data, alpha, epsilon)
    return model