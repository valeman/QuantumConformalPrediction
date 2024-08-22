import torch
import torch.nn.functional as F

import torchquantum as tq
import numpy as np
from scipy.stats import norm
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from customDistribution import combinedNormals

from torch.utils.data import DataLoader, TensorDataset
from helper_functions import evenlySpaceEigenstates, toClosestEigenstate




class NegativeLogSumCriterion(nn.Module):
    def __init__(self):
        super(NegativeLogSumCriterion, self).__init__()

    def forward(self, input_tensor):
        clamped_tensor = torch.clamp(input_tensor, min=0.01)
        log_tensor = torch.log(clamped_tensor)
        sum_log = torch.sum(log_tensor)
        return -sum_log


class QuantumLayer(tq.QuantumModule):
    def __init__(self, n_wires, n_layers):
        super(QuantumLayer, self).__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=1)  # Batch size set to 1 for simplicity
        self.n_layers = n_layers
        self.n_wires = n_wires

        self.rz_layers1 = torch.nn.ModuleList([
            tq.RZ(has_params=True, trainable=True) for _ in range(n_layers * n_wires)
        ])
        self.ry_layers = torch.nn.ModuleList([
            tq.RY(has_params=True, trainable=True) for _ in range(n_layers * n_wires)
        ])
        self.rz_layers2 = torch.nn.ModuleList([
            tq.RZ(has_params=True, trainable=True) for _ in range(n_layers * n_wires)
        ])
        self.cz = tq.CZ(has_params=False, trainable=False)


    @tq.static_support
    def forward(self, q_device=None, measure=False):

        q_device = self.q_device
        q_device.reset_states(1)

        for l in range(self.n_layers):
            for wire in range(self.n_wires):
                # Apply the parameterized RY and RZ gates to each wire
                self.rz_layers1[l * self.n_wires + wire](q_device, wires=wire)
                self.ry_layers[l * self.n_wires + wire](q_device, wires=wire) 
                self.rz_layers2[l * self.n_wires + wire](q_device, wires=wire)
            
            # Apply CZ gates between all pairs of qubits
            for i in range(self.n_wires):
                for j in range(i + 1, self.n_wires):
                    self.cz(q_device, wires=[i, j])

        if measure: return tq.measurements.measure(q_device, n_shots=1024)
        else: return q_device.get_states_1d()


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
n_layers = 3
n_epochs = 10

model = QuantumLayer(n_wires=n_wires, n_layers=n_layers)

# Apply the quantum circuit to the QuantumDevice
criterion = NegativeLogSumCriterion()
optimizer = optim.Adam(model.parameters(), lr=0.01)

dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
training_samples = toClosestEigenstate(torch.from_numpy(dist.rvs(size=50)), n_wires, -1, 1)
dataset = TensorDataset(training_samples)
data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
losses = []

# for epoch in range(n_epochs):
#     optimizer.zero_grad()
#     for batch_samples in data_loader:  
#         output = model()
#         model_probabilities = calculate_probabilities(output, batch_samples[0])
#         true_probabilities = dist.pdf_list(evenlySpaceEigenstates(batch_samples[0], n_wires, -1, 1))
#         loss = criterion(model_probabilities)
#         losses.append(loss.item())
#         loss.backward()

#         optimizer.step()
#     print(epoch)

print(model)



# Plotting the loss over time
# plt.plot(range(n_epochs*10), losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss over Time')
# plt.show()

# data = model(measure=True)[0]

# Extract keys and values
# states = [evenlySpaceEigenstates(bitstring, n_wires, -1, 1) for bitstring in data.keys()]
# counts = list(data.values())

# x_values = np.linspace(-1, 1, 1000)

# pdf_values = 0.5*(norm.pdf(x_values, loc=-0.75, scale=0.1) + norm.pdf(x_values, loc=0.75, scale=0.1))
# fig, ax1 = plt.subplots()

# ax1.bar(states, counts)
# ax2 = ax1.twinx()
# ax2.plot(x_values, pdf_values, color='g', label="PDF of Sum of Normals")

# plt.show()