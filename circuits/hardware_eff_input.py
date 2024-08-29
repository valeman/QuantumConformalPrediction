import torchquantum as tq
import torch
from circuits.angle_encodings import *

class HardwareEfficientNoInput(tq.QuantumModule):
    def __init__(self, n_wires, n_layers):
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=1)
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.rz_layers1 = tq.QuantumModuleList()
        self.ry_layers = tq.QuantumModuleList()
        self.rz_layers2 = tq.QuantumModuleList()
        self.cnot_layers = tq.QuantumModuleList()

        for _ in range(n_layers):
            self.rz_layers1.append(
                tq.Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.ry_layers.append(
                tq.Op1QAllLayer(
                    op=tq.RY,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.rz_layers2.append(
                tq.Op1QAllLayer(
                    op=tq.RZ,
                    n_wires=n_wires,
                    has_params=True,
                    trainable=True,
                )
            )
            self.cnot_layers.append(
                tq.Op2QAllLayer(
                    op=tq.CZ,
                    n_wires=n_wires,
                    has_params=False,
                    trainable=False,
                    circular=False,
                )
            )
    def forward(self, q_device=None):
        q_device = self.q_device if q_device == None else q_device
        q_device.reset_states(1)
        for k in range(self.n_layers):
            self.rz_layers1[k](q_device)
            self.ry_layers[k](q_device)
            self.rz_layers2[k](q_device)
            self.cnot_layers[k](q_device)
        

    def calculate_probabilities(self, target_eigenvectors_denary):
        """
        Calculates the probabilities of target eigenvectors from the current statevector.

        Parameters:
        - target_eigenvectors_denary (torch.Tensor): A tensor containing the indices of target eigenvectors in denary.

        Returns:
        - torch.Tensor: A tensor containing the probabilities corresponding to the target eigenvectors.
        """
        # Get the current statevector
        self.forward()
        statevector = self.q_device.get_states_1d()
        # Flatten the statevector and calculate the probabilities
        target_eigenvectors_denary = target_eigenvectors_denary.to(torch.int64)
        statevector = statevector.flatten()
        probability_amplitudes = statevector[target_eigenvectors_denary]  # PyTorch fancy indexing
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
            
 

    def sample_from_model(self, n_shots):
        """
        Samples from the quantum device after running the quantum circuit.

        Parameters:
        - n_shots (int): The number of samples (shots) to take.

        Returns:
        - torch.Tensor: A tensor containing the sampled measurement outcomes.
        """
        # Run the forward method with measurement
        self.forward()
        samples = tq.measurements.measure(self.q_device, n_shots=n_shots)
        return samples