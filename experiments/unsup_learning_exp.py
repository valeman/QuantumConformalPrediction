import torchquantum as tq
from training import UnsupDensityModelTrainer
from models import HardwareEfficientNoInput
from utils import combinedNormals
from utils.helper_functions import toClosestEigenstate, evenlySpaceEigenstates
from torch.utils.data import TensorDataset
import torch
from torchquantum.plugin import tq2qiskit
from qiskit.visualization import circuit_drawer
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity




n_wires = 5
n_layers = 2
n_epochs = 50
batch_size = 20
n_training_samples = 50

q_device = tq.QuantumDevice(n_wires=n_wires)

circuit = HardwareEfficientNoInput(n_wires=n_wires, n_layers=n_layers)
dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
training_samples = toClosestEigenstate(torch.from_numpy(dist.rvs(size=n_training_samples)), n_wires, -1.5, 1.5)
dataset = TensorDataset(training_samples)
trainer = UnsupDensityModelTrainer(circuit, q_device, dataset, batch_size, n_epochs)
trainedCircuit = trainer.train()
qiskit_circuit = tq2qiskit(q_device, trainedCircuit)

circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})

plt.show()


# Plotting the loss over time
plt.plot(range(n_epochs), losses)
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