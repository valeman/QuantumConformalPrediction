from qiskit import qpy
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt
from CPProcedure.sample_on_devices import *

devices = {
    "IBMQ": sample_with_ibmq,
    "IBMQ M3": sample_with_ibmqM3,
    "aer": sample_with_qiskit_aer,
}

class ConformalPredictionProcedure:

    def __init__(self, callibration_data, circuit_file_name, device, nc_score_function, alpha, M):
        self.callibration_data = callibration_data
        self.circuit = self.loadCircuit(circuit_file_name)
        self.device = device
        self.nc_score_function = nc_score_function
        self.alpha = alpha
        self.M = M

    def loadCircuit(self, file_name):
        """
        retrieve and return trained qiskit circuit reverse the bits (to match torchquantum implementation)
        """
        with open("./circuits/savedQiskitCircuits/" + file_name, 'rb') as handle:
            qc = qpy.load(handle)[0]
            qc = qc.reverse_bits()
         
        circuit_drawer(qc, output='mpl', style={'name': 'bw'})
        plt.show()
        return qc

    def runTrainedModel(self):
        sammple_with_device = devices.get(self.device)
        sammple_with_device(self.circuit, self.M)

    def computeQuantile(self):
        pass

    def generatePredictionSet(self):
        pass

    def score():
        pass