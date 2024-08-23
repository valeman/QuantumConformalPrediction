import torchquantum as tq
import torch
import torch.nn as nn
from models.base_model import BaseModel

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
    def forward(self, q_device=None, measure=False, reset_states=True):
        if q_device == None: q_device = self.q_device
        if reset_states: q_device.reset_states(1)

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

        if measure: return tq.measurements.measure(q_device, n_shots=1024)
        else: return q_device.get_states_1d()


