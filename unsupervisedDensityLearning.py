import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.quantum_info as qi
from qiskit.circuit.library import TwoLocal
import pickle

NUM_LAYERS = 2
NUM_QUBITS = 5

# should I split the data to make Dtr or DCal or should I sample from it.
def generateDataset(datasetSize):
    choices = np.random.choice([0, 1], size=datasetSize, p=[0.5, 0.5])

    samples = np.where(
        choices == 0, 
        np.random.normal(-0.75, 0.1, datasetSize),
        np.random.normal(0.75, 0.1, datasetSize)
    )
    
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(1)

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
        pickle.dump(bound_circuit, file)

def loadCircuit(name):
    with open(name + '.pkl', 'rb') as file:
        return pickle.load(file)
    
def bitstring_to_continuous_value(bitstring, num_qubits):
    j = int(bitstring, 2)  # Convert bitstring to an integer index
    N = 2**num_qubits  # Total number of possible bitstrings
    y_j = -1 + 2 * (j - 1) / (N - 1)
    return y_j

class QuantumCircuitModel(nn.Module):
    def __init__(self, circuit, simulator, num_qubits):
        super(QuantumCircuitModel, self).__init__()
        self.circuit = circuit
        self.simulator = simulator
        self.num_qubits = num_qubits
        self.params = nn.Parameter(torch.tensor(random_params, dtype=torch.float32))

    def forward(self):
        bound_circuit = self.circuit.assign_parameters(self.params.detach().numpy())
        transpiled_circuit = transpile(bound_circuit, self.simulator)
        result = self.simulator.run(transpiled_circuit, shots=100).result()
        memory = result.get_memory(transpiled_circuit)
        
        # Convert measurement results to continuous values
        continuous_values = torch.tensor([bitstring_to_continuous_value(meas, self.num_qubits) for meas in memory], dtype=torch.float32)
        return continuous_values



callibration_data = generateDataset(10)
training_data = generateDataset(10)

# Create circuit


# Transpile for simulator
simulator = AerSimulator()
circuit = TwoLocal(NUM_QUBITS, ['rz','ry','rz'], 'cz', 'linear', reps=NUM_LAYERS, insert_barriers=True) 
print(circuit.decompose().draw())

# Generate random parameters within the bounds [-2π, 2π]
# Check this is correct later on
random_params = np.random.uniform(low=-2*np.pi, high=2*np.pi, size=circuit.num_parameters)
circuit.measure_all()
circuit = circuit.assign_parameters(random_params)



# Initialize the model
model = QuantumCircuitModel(circuit, simulator, NUM_QUBITS)

# Define the loss function (e.g., Mean Squared Error)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):  # Number of epochs
    optimizer.zero_grad()
    
    # Forward pass: get continuous values from the model
    outputs = model()
    
    # Calculate the target distribution values
    pdf_values = 0.5 * (norm.pdf(outputs.numpy(), loc=-0.75, scale=0.1) + norm.pdf(outputs.numpy(), loc=0.75, scale=0.1))
    
    # Convert pdf_values to a PyTorch tensor
    targets = torch.tensor(pdf_values, dtype=torch.float32)
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch + 1}/100, Loss: {loss.item()}')
