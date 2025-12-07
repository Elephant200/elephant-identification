import numpy as np
from scipy.special import logsumexp

def combine_results(results1: np.ndarray, results2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    """
    Combine two results array using log-opinion pooling.

    Args:
        results1 (np.ndarray): The first results ndarray of shape (n_samples, n_classes), where each row is a sample and each column is a class. The values are the probabilities of the sample being in the class.
        results2 (np.ndarray): The second results ndarray of shape (n_samples, n_classes), where each row is a sample and each column is a class. The values are the probabilities of the sample being in the class.
        w1 (float): The weight of the first results, like top-1 accuracy of model 1
        w2 (float): The weight of the second results, like top-1 accuracy of model 2

    Returns:
        np.ndarray: The combined results ndarray of shape (n_samples, n_classes)
    """
    eps = 1e-12

    weight_sum = w1 + w2
    w1 = w1 / weight_sum
    w2 = w2 / weight_sum

    logP = w1 * np.log(results1 + eps) + w2 * np.log(results2 + eps)
    logP = logP - logsumexp(logP, axis=1, keepdims=True)
    return np.exp(logP)

if __name__ == "__main__":
    results1 = np.array([
        [0.1, 0.2, 0.3], 
        [0.4, 0.5, 0.6]
        ])
    results2 = np.array([
        [0.2, 0.2, 0.6], 
        [0.4, 0.2, 0.4]
        ])
    w1 = 0.3
    w2 = 0.6
    combined_results = combine_results(results1, results2, w1, w2)
    print(combined_results)
