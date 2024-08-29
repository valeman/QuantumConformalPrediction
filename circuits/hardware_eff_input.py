import torchquantum as tq
import torch
from circuits.angle_encodings import *
import numpy as np
import torchquantum.functional as tqf
from utils.helper_functions import evenlySpaceEigenstates


class HardwareEfficientInput(tq.QuantumModule):
    def __init__(self, n_wires, n_layers, bsz):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=bsz)
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.bsz = bsz
        self.angle_encoding = LearnedNonLinear(n_layers, n_wires)
        self.cz_layers = tq.QuantumModuleList()

        for k in range(n_layers):
            self.cz_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CZ,
                    n_wires=n_wires,
                    has_params=False,
                    trainable=False,
                    circular=False,
                )
            )

    tq.static_support
    def forward(self, x, q_device=None):
        q_device = self.q_device if q_device is None else q_device
        q_device.reset_states(x.size(0))

        # Apply angle encoding to get the parameters for the gates
        angles = self.angle_encoding(x.float())
        # Reshape angles to (batch_size, n_layers, n_wires, 3)
        angles = angles.view(x.size(0), self.n_layers, self.n_wires, 3)

        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                # Apply the rz and ry gates in parallel for the whole batch
                tqf.rz(q_device, wires=wire, params=angles[:, layer, wire, 0], static=self.static_mode, parent_graph=self.graph)
                tqf.ry(q_device, wires=wire, params=angles[:, layer, wire, 1], static=self.static_mode, parent_graph=self.graph)
                tqf.rz(q_device, wires=wire, params=angles[:, layer, wire, 2], static=self.static_mode, parent_graph=self.graph)
            
            # Apply the CZ layer for this layer
            self.cz_layers[layer](q_device)


    def calculate_probabilities(self, input, target_eigenvectors_denary):
        """
        Calculates the probabilities of target eigenvectors from the current statevector.

        Parameters:
        - target_eigenvectors_denary (torch.Tensor): A tensor containing the indices of target eigenvectors in denary.

        Returns:
        - torch.Tensor: A tensor containing the probabilities corresponding to the target eigenvectors.
        """
        # Get the current statevector
        self.forward(input)
        statevector = self.q_device.get_states_1d()
        # print("STATE", statevector)
        # Flatten the statevector and calculate the probabilities
        target_eigenvectors_denary = target_eigenvectors_denary.to(torch.int64)
        # print("TARGETS", target_eigenvectors_denary)
        # statevector = statevector.flatten()

        row_indices = torch.arange(len(target_eigenvectors_denary))

        probability_amplitudes = statevector[row_indices, target_eigenvectors_denary]

        return probability_amplitudes.abs() ** 2

    def calculate_expected_value(self, eigenvalues):
        """
        Calculates the expectation value of an observable given its eigenvalues.

        Parameters:
        - eigenvalues (torch.Tensor): The eigenvalues of the observable.

        Returns:
        - torch.Tensor: The expectation value.
        """
        # Prepare the quantum state without measurement
        self.forward()
        statevector = self.q_device.get_states_1d().flatten()
        return torch.dot(statevector.abs()**2, eigenvalues)
            
 

    def sample_from_model(self, x_data):
        """
        Samples from the quantum device after running the quantum circuit.

        Parameters:
        - n_shots (int): The number of samples (shots) to take.

        Returns:
        - torch.Tensor: A tensor containing the sampled measurement outcomes.
        """
        # Run the forward method with measurement
        expanded_samples = []
        for x in x_data:
            self.forward(x.unsqueeze(0))
            samples = tq.measurements.measure(self.q_device, n_shots=1)
            for key, freq in samples[0].items():
                value = evenlySpaceEigenstates(key, self.n_wires, -2, 2)
                expanded_samples.extend([value] * freq)
        return expanded_samples