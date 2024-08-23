import torch
def calculate_probabilities(statevector, target_eigenvectors_denary):
    """
    Calculates the probabilities of target eigenvectors from a given statevector.

    Parameters:
    - statevector (torch.Tensor): The state vector from which to calculate probabilities.
    - target_eigenvectors_denary (torch.Tensor): A tensor containing the indices of target eigenvectors in denary.

    Returns:
    - torch.Tensor: A tensor containing the probabilities corresponding to the target eigenvectors.
    """
    target_eigenvectors_denary = target_eigenvectors_denary.to(torch.int64)
    statevector = statevector.flatten()
    probability_amplitudes = statevector[target_eigenvectors_denary] # pytorch fancy indexing
    return probability_amplitudes.abs() ** 2