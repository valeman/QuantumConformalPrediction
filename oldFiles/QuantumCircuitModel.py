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
        continuous_values = torch.tensor([bitstringToContinuousValue(meas, self.num_qubits, -1, 1) for meas in memory], dtype=torch.float32)
        return continuous_values