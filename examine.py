import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
    tq2qiskit
)

qdev = tq.QuantumDevice(n_wires=2, bsz=5, device="cpu", record_op=True) # use device='cuda' for GPU
# use qdev.op
qdev.h(wires=0)
qdev.cnot(wires=[0, 1])

# use tqf
tqf.h(qdev, wires=1)
tqf.x(qdev, wires=1)

# use tq.Operator
op = tq.RX(has_params=True, trainable=True, init_params=0.5)
op(qdev, wires=0)
print(qdev)
measurement_circ = tq2qiskit(qdev)


# print the current state (dynamic computation graph supported)