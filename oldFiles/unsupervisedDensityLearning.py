import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import random
import math
from qiskit.quantum_info import DensityMatrix, Statevector


# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer import Aer
import qiskit.quantum_info as qi
from qiskit.circuit.library import TwoLocal
import pickle

from qiskit_aer.library import SaveDensityMatrix
from utils.combined_normals import combinedNormals

from torch.utils.data import DataLoader, TensorDataset


NUM_LAYERS = 2
NUM_QUBITS = 5





def plotSamplesAndDistribution(training_data, callibration_data):

    min_data_point = min(-1.5, min(training_data), min(callibration_data))
    max_data_point = max(1.5, max(training_data), min(callibration_data))
    x_values = np.linspace(min_data_point, max_data_point, 1000)

    pdf_values = 0.5*(norm.pdf(x_values, loc=-0.75, scale=0.1) + norm.pdf(x_values, loc=0.75, scale=0.1))

    plt.plot(x_values, pdf_values, color='g', label="PDF of Sum of Normals")

    plt.plot(training_data.numpy(), np.zeros_like(training_data.numpy()), 'gx', label="Training Data")
    plt.plot(callibration_data.numpy(), np.zeros_like(callibration_data.numpy()), 'o', markerfacecolor='none', color='blue',label="Calibration Data")

    # Labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Samples vs PDF of Distribution')
    plt.legend()

    # Show the plot
    plt.show()

def saveCircuit(circuit, name):
    with open(name + '.pkl', 'wb') as file:
        pickle.dump(circuit, file)

def loadCircuit(name):
    with open(name + '.pkl', 'rb') as file:
        return pickle.load(file)
    
def bitstringToContinuousValue(bitstring, num_qubits, min, max):
    j = int(bitstring, 2)  # Convert bitstring to an integer index
    N = 2**num_qubits  # Total number of possible bitstrings
    y_j = min + (j - 1) * (max-min) / (N - 1)
    return y_j

def samplesToDenaryEigenvectors(tensor, num_qubits, min_val, max_val):
    return torch.round((tensor - min_val) * (2**num_qubits - 1) / (max_val - min_val))

def denaryEigenvectorsToOutputSpace(tensor, num_qubits, min_val, max_val):
    return min_val + tensor * (max_val - min_val) / (2**num_qubits - 1)



    
class QuantumCircuitModel(nn.Module):
    def __init__(self, circuit : QuantumCircuit, simulator : AerSimulator, num_qubits):
        super(QuantumCircuitModel, self).__init__()
        self.circuit = circuit
        self.simulator = simulator
        self.num_qubits = num_qubits
        self.params = nn.Parameter(torch.randn(circuit.num_parameters, dtype=torch.float32))


    def forward(self):
        # Update circuit parameters
        bound_circuit = self.circuit.assign_parameters(self.params.detach().numpy())
        transpiled_circuit = transpile(bound_circuit, self.simulator)
        density_matrix = self.simulator.run(transpiled_circuit).result().data()['density_matrix']
        return density_matrix
    
    # def probOfOutputs(self, density_matrix, true_labels):
    #     probabilities = []
    #     for i in range(true_labels.size(0)):
    #         proj_op = true_labels[i].to_operator()
    #         proj_matrix = torch.matmul(proj_op, density_matrix[i])
    #         prob = torch.trace(proj_matrix).real
    #         probabilities.append(prob)
    #     return torch.stack(probabilities)
    
    def probOfOutputs(self, density_matrix, true_points):

        probabilities = torch.zeros(true_points.size())
        true_points_int = samplesToDenaryEigenvectors(true_points, NUM_QUBITS, -1, 1)

        print(true_points_int)
        print(2**NUM_QUBITS)
        for i in range(true_points_int.size(0)):
            statevector = Statevector.from_int(int(true_points_int[i].item()), 2**NUM_QUBITS)
            projector = np.outer(statevector.data, statevector.data.conj())
            probabilities[i] = np.trace(density_matrix.data @ projector).real

        return probabilities




    




simulator = AerSimulator(method='density_matrix')
circuit = TwoLocal(NUM_QUBITS, ['rz','ry','rz'], 'cz', 'linear', reps=NUM_LAYERS) 
save_density_matrix = SaveDensityMatrix(circuit.num_qubits, label="density_matrix")
circuit.append(save_density_matrix, circuit.qubits)

# param_vector = ParameterVector('θ', length=circuit.num_parameters)
# circuit = circuit.assign_parameters(param_vector)

# Initialize the model
model = QuantumCircuitModel(circuit, simulator, NUM_QUBITS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# create training data
dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
training_samples = dist.rvs(size=50)
# training_eigenvectors = samplesToDenaryEigenvectors(training_samples, NUM_QUBITS, -1, 1)
# print(training_eigenvectors)
# quantized_samples = denaryEigenvectorsToOutputSpace(training_eigenvectors, NUM_QUBITS, -1, 1)

# bound_circuit = circuit.assign_parameters([random.uniform(0, 2 * math.pi) for i in range(circuit.num_parameters)])
# print(bound_circuit.decompose().draw())
# transpiled_circuit = transpile(bound_circuit, simulator)
# data = simulator.run(transpiled_circuit).result().data()
# print(data['density_matrix'])

# # Training loop
dataset = TensorDataset(training_samples)
data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
for epoch in range(1):  # iterate over epochs
    optimizer.zero_grad()

    for batch_samples in data_loader:  
        output = model()

        loss = criterion(model.probOfOutputs(output, batch_samples[0]), dist.pdf_list(denaryEigenvectorsToOutputSpace(batch_samples[0], NUM_QUBITS, -1, 1)))
            
        loss.backward()

        optimizer.step()
    
    # print(f'Epoch {epoch + 1}/100, Loss: {loss.item()}')
