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
def selective_pruning_layer_rms(model, layer_index, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    RMS strategy: Calculate the root mean square activation of neurons.
    For each neuron n:
       I_rms(D, n) = sqrt(mean(z(d)^2)).
    Score = I_rms(D_forget, n) / (I_rms(D_retain, n) + epsilon).
    Prune neurons with the highest scores.
    """
    X_good = retain_data[0]
    X_bad = forget_data[0]
    for it in range(num_iter):
        acts_good = model.get_activation(X_good, layer_index).detach().cpu().numpy()
        acts_bad = model.get_activation(X_bad, layer_index).detach().cpu().numpy()
        num_neurons = acts_good.shape[1]
        scores = np.zeros(num_neurons)
        for n in range(num_neurons):
            I_retain = np.sqrt(np.mean(acts_good[:, n]**2))
            I_forget = np.sqrt(np.mean(acts_bad[:, n]**2))
            scores[n] = I_forget / (I_retain + epsilon)
        k = int(alpha * num_neurons)
        prune_idx = np.argsort(scores)[-k:]
        print(f"RMS strategy: Layer {layer_index} iter {it}: prune neurons {prune_idx}, scores {scores[prune_idx]}")
        for idx in prune_idx:
            prune_neuron(model, layer_index, idx)
    return model

def selective_pruning_multi_layers_rms(model, layer_indices, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    Apply RMS pruning strategy to multiple specified layers.
    """
    for li in layer_indices:
        model = selective_pruning_layer_rms(model, li, retain_data, forget_data, alpha, num_iter, epsilon)
    return model

def selective_pruning_layer_freq(model, layer_index, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    FREQ strategy: Calculate the activation frequency of each neuron in the dataset.
    Calculate activation frequencies separately on retain and forget data, with score:
         score = I_freq(D_forget, n) / (I_freq(D_retain, n) + epsilon)
    Prune neurons with the highest scores.
    """
    X_good = retain_data[0]
    X_bad = forget_data[0]
    for it in range(num_iter):
        acts_good = model.get_activation(X_good, layer_index).detach().cpu().numpy()  # shape: [N_good, num_neurons]
        acts_bad = model.get_activation(X_bad, layer_index).detach().cpu().numpy()    # shape: [N_bad, num_neurons]
        num_neurons = acts_good.shape[1]
        scores = np.zeros(num_neurons)
        for n in range(num_neurons):
            # Calculate activation frequency, i.e., proportion of positive values
            I_retain = np.sum(acts_good[:, n] > 0) / acts_good.shape[0]
            I_forget = np.sum(acts_bad[:, n] > 0) / acts_bad.shape[0]
            scores[n] = I_forget / (I_retain + epsilon)
        k = int(alpha * num_neurons)
        # Select the k neurons with the highest scores
        prune_idx = np.argsort(scores)[-k:]
        print(f"FREQ strategy: Layer {layer_index} iter {it}: prune neurons {prune_idx}, scores {scores[prune_idx]}")
        for idx in prune_idx:
            prune_neuron(model, layer_index, idx)
    return model

def selective_pruning_multi_layers_freq(model, layer_indices, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    Apply FREQ pruning strategy to multiple specified layers.
    """
    for li in layer_indices:
        model = selective_pruning_layer_freq(model, li, retain_data, forget_data, alpha, num_iter, epsilon)
    return model

def selective_pruning_layer_std(model, layer_index, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    STD strategy: Calculate the standard deviation of neuron activations.
    For each neuron n:
       I_std(D, n) = std(z(d)).
    Score = I_std(D_forget, n) / (I_std(D_retain, n) + epsilon).
    Prune neurons with the highest scores.
    """
    X_good = retain_data[0]
    X_bad = forget_data[0]
    for it in range(num_iter):
        acts_good = model.get_activation(X_good, layer_index).detach().cpu().numpy()
        acts_bad = model.get_activation(X_bad, layer_index).detach().cpu().numpy()
        num_neurons = acts_good.shape[1]
        scores = np.zeros(num_neurons)
        for n in range(num_neurons):
            I_retain = np.std(acts_good[:, n])
            I_forget = np.std(acts_bad[:, n])
            scores[n] = I_forget / (I_retain + epsilon)
        k = int(alpha * num_neurons)
        prune_idx = np.argsort(scores)[-k:]
        print(f"STD strategy: Layer {layer_index} iter {it}: prune neurons {prune_idx}, scores {scores[prune_idx]}")
        for idx in prune_idx:
            prune_neuron(model, layer_index, idx)
    return model

def selective_pruning_multi_layers_std(model, layer_indices, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    Apply STD pruning strategy to multiple specified layers.
    """
    for li in layer_indices:
        model = selective_pruning_layer_std(model, li, retain_data, forget_data, alpha, num_iter, epsilon)
    return model

def selective_pruning_layer_abs(model, layer_index, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    ABS strategy: Calculate the mean absolute activation of neurons.
    For each neuron n:
       I_abs(D, n) = average(|z(d)|) over D.
    Score = I_abs(D_forget, n) / (I_abs(D_retain, n) + epsilon).
    Prune neurons with the highest scores.
    """
    X_good = retain_data[0]
    X_bad = forget_data[0]
    for it in range(num_iter):
        acts_good = model.get_activation(X_good, layer_index).detach().cpu().numpy()
        acts_bad = model.get_activation(X_bad, layer_index).detach().cpu().numpy()
        num_neurons = acts_good.shape[1]
        scores = np.zeros(num_neurons)
        for n in range(num_neurons):
            I_retain = np.mean(np.abs(acts_good[:, n]))
            I_forget = np.mean(np.abs(acts_bad[:, n]))
            scores[n] = I_forget / (I_retain + epsilon)
        k = int(alpha * num_neurons)
        prune_idx = np.argsort(scores)[-k:]
        print(f"ABS strategy: Layer {layer_index} iter {it}: prune neurons {prune_idx}, scores {scores[prune_idx]}")
        for idx in prune_idx:
            prune_neuron(model, layer_index, idx)
    return model

def selective_pruning_multi_layers_abs(model, layer_indices, retain_data, forget_data, alpha=0.01, num_iter=3, epsilon=1e-8):
    """
    Apply ABS pruning strategy to multiple specified layers.
    """
    for li in layer_indices:
        model = selective_pruning_layer_abs(model, li, retain_data, forget_data, alpha, num_iter, epsilon)
    return model
