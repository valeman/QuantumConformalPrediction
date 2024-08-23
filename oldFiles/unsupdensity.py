import torch
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
from scipy.stats import norm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.customDistribution import combinedNormals

from torch.utils.data import DataLoader, TensorDataset
from utils.helper_functions import evenlySpaceEigenstates, toClosestEigenstate

from torchquantum.plugin import tq2qiskit
from qiskit.visualization import circuit_drawer
from sklearn.neighbors import KernelDensity
from circuits.hardware_eff_no_input import HardwareEfficientNoInput
from training.metrics import NegativeLogSumCriterion

import torch.nn as nn






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

# Define your parameters
n_wires = 5
n_layers = 2
n_epochs = 50
batch_size = 20
n_training_samples = 50

q_device = tq.QuantumDevice(n_wires=n_wires)
circuit = HardwareEfficientNoInput(n_wires=n_wires, n_layers=n_layers)

criterion = NegativeLogSumCriterion()
optimizer = optim.Adam(circuit.parameters(), lr=0.01)

dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
training_samples = toClosestEigenstate(torch.from_numpy(dist.rvs(size=n_training_samples)), n_wires, -1.5, 1.5)
dataset = TensorDataset(training_samples)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
losses = []
loss = 0

for epoch in range(n_epochs):
    optimizer.zero_grad()
    for batch_samples in data_loader:  
        output = circuit(q_device=q_device)
        model_probabilities = calculate_probabilities(output, batch_samples[0])
        true_probabilities = dist.pdf_list(evenlySpaceEigenstates(batch_samples[0], n_wires, -1.5, 1.5))
        loss = criterion(model_probabilities)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())
        print(epoch)

qiskit_circuit = tq2qiskit(q_device, circuit)
circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})

plt.show()


# Plotting the loss over time
plt.plot(range(150), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Time')
plt.show()

measurements = circuit(q_device=q_device, measure=True)
data = measurements[0]

expanded_data = []
for key, freq in data.items():
    value = evenlySpaceEigenstates(key, n_wires, -1.5, 1.5)
    expanded_data.extend([value] * freq)

# Convert to 2D array for KDE
expanded_data = np.array(expanded_data).reshape(-1, 1)

# Extract keys and values
states = [evenlySpaceEigenstates(bitstring, n_wires, -1.5, 1.5) for bitstring in data.keys()]
counts = list(data.values())
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(np.array(expanded_data).reshape(-1, 1))


x_values = np.linspace(-1, 1, 1000)

pdf_values = 0.5*(norm.pdf(x_values, loc=-0.75, scale=0.1) + norm.pdf(x_values, loc=0.75, scale=0.1))
kde_values = np.exp(kde.score_samples(x_values.reshape(-1, 1)))

fig, ax1 = plt.subplots()

ax1.bar(states, counts)
ax2 = ax1.twinx()
ax2.plot(x_values, pdf_values, color='g', label="PDF of Sum of Normals")
ax2.plot(x_values, kde_values, color="r", label="kde")

plt.show()