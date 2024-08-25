from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
import matplotlib.pyplot as plt

def sample_with_ibmq(qc, n_shots):
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    ideal_distribution = Statevector.from_instruction(qc).probabilities_dict()
    
    # add qubit measurements to the circuit
    qc.measure_all()

    # transpile the circuit
    pass_manager = generate_preset_pass_manager(3, backend=backend, seed_transpiler=0)
    isa_qc = pass_manager.run(qc)
    
    # run sampler
    print("running measurements on:", backend.name)    
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_qc], shots=n_shots)
    print(f">>> Job ID: {job.job_id()}")
    print(f">>> Job Status: {job.status()}")
    result = job.result()
    
    # plot histogram of samples against ideal distribution
    binary_prob = [{k: v / res.data.meas.num_shots for k, v in res.data.meas.get_counts().items()} for res in result]
    plot_histogram(
    binary_prob + [ideal_distribution],
    bar_labels=False,
    legend=[
        "optimization_level=3",
        "ideal distribution",
    ],)
    plt.show()

    return result[0].data.meas.get_counts()

def sample_with_ibmqM3(circuit, n_samples):
    pass

def sample_with_qiskit_aer(circuit, n_samples):
    pass