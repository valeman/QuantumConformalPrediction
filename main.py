# from experiments.unsup_learning_exp import train_and_save_model
import torchquantum as tq
import matplotlib.pyplot as plt
import torch
from circuits.hardware_eff_input import HardwareEfficientInput
from torchquantum.plugin import (
    tq2qiskit_initialize,
    tq2qiskit,
    tq2qiskit_measurement,
    qiskit_assemble_circs,
)

from experiments.unsup_learning_exp import train_and_save_model
train_and_save_model("tester.qpy", "IP")

# q_device = tq.QuantumDevice(n_wires=3)


# circuit = HardwareEfficientInput(n_wires=3, n_layers=2)

# circuit(q_device)
# print(circuit)
# qiskit_circuit = tq2qiskit(q_device, circuit)
# print(qiskit_circuit)