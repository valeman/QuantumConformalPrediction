import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt
from qiskit_aer.library import SaveDensityMatrix


# Create circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
save_density_matrix = SaveDensityMatrix(2, label="density_matrix")
circ.append(save_density_matrix, circ.qubits)

# circ.measure_all()

simulator = AerSimulator(method='density_matrix')
circ_t = transpile(circ, simulator)
data = simulator.run(circ_t).result().data()
print(data['density_matrix'])
# plot_histogram(counts, title='Bell-State counts')
# plt.show()

print(circ_t.decompose().draw())