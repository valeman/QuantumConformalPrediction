import torchquantum as tq
from circuits.base_model import BaseModel
import torch
from torchquantum.measurement import expval_joint_analytical

class HardwareEfficientNoInput(tq.QuantumModule, BaseModel):
    def __init__(self, n_wires, n_layers):
        super(HardwareEfficientNoInput, self).__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires, bsz=1)  # Batch size set to 1 for simplicity
        self.n_layers = n_layers
        self.n_wires = n_wires

        for layer in range(n_layers):
            for wire in range(n_wires):
                setattr(self, f"rz1_layer{layer}_wire{wire}", tq.RZ(has_params=True, trainable=True))
                setattr(self, f"ry_layer{layer}_wire{wire}", tq.RY(has_params=True, trainable=True))
                setattr(self, f"rz2_layer{layer}_wire{wire}", tq.RZ(has_params=True, trainable=True))

            for wire in range(n_wires - 1):
                setattr(self, f"cz_layer{layer}_wires{wire}_{wire + 1}", tq.CZ(has_params=False, trainable=False))

        self.cz = tq.CZ(has_params=False, trainable=False)

    @tq.static_support
    def forward(self, q_device=None, reset_states=True):
        if q_device is None:
            q_device = self.q_device
        if reset_states:
            q_device.reset_states(1)

        for l in range(self.n_layers):
            for wire in range(self.n_wires):
                # Apply the parameterized RY and RZ gates to each wire
                rz1_gate = getattr(self, f"rz1_layer{l}_wire{wire}")
                ry_gate = getattr(self, f"ry_layer{l}_wire{wire}")
                rz2_gate = getattr(self, f"rz2_layer{l}_wire{wire}")

                rz1_gate(q_device, wires=wire)
                ry_gate(q_device, wires=wire)
                rz2_gate(q_device, wires=wire)

            for wire in range(self.n_wires - 1):
                cz_gate = getattr(self, f"cz_layer{l}_wires{wire}_{wire + 1}")
                cz_gate(q_device, wires=[wire, wire + 1])

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

        # Define the observable
        pauli_operators = [tq.PauliZ(wires=0)]
        observable = tq.Observable(eigenvalues=eigenvalues, 
                            observables=pauli_operators,
                            has_params=False, 
                            trainable=False) 
        # Calculate the expectation value
        expectation_value = self.q_device.expval_joint_analytical(observable)
        return expectation_value

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
