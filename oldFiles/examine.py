import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
    tq2qiskit
)

import utils.helper_functions as helper_functions
from utils.customDistribution import combinedNormals

x = helper_functions.toClosestEigenstate(1, 5, -1, 1)

print(x)