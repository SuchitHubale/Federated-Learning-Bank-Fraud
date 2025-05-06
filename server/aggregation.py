import torch
import numpy as np

def fed_avg(weights_list):
    """
    Perform Federated Averaging on a list of model weights.
    Args:
        weights_list: List of lists of numpy arrays (weights from each client)
    Returns:
        Averaged weights as a list of numpy arrays
    """
    avg_weights = []
    for weights in zip(*weights_list):
        avg = np.mean(np.array(weights), axis=0)
        avg_weights.append(avg)
    return avg_weights 