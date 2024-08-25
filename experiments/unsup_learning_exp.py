from qiskit import qpy
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer



N_WIRES = 5
N_LAYERS = 2
N_EPOCHS = 70
BATCH_SIZE = 20
N_TRAINING_SAMPLES = 50

# must be in qtvenv
def train_and_save_model(save_pqc_file_name, plot_results=False):
    import torchquantum as tq
    from torchquantum.plugin import tq2qiskit 
    from torch.utils.data import TensorDataset
    import torch
    from training.unsup_density_trainer import BackpropogationTrainer
    from circuits import HardwareEfficientNoInput
    from utils import combinedNormals
    from utils.helper_functions import toClosestEigenstate 

    # create training data
    dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
    training_samples = toClosestEigenstate(torch.from_numpy(dist.rvs(size=N_TRAINING_SAMPLES)), N_WIRES, -1.5, 1.5)
    dataset = TensorDataset(training_samples)

    q_device = tq.QuantumDevice(n_wires=N_WIRES)

    # create and train pqc
    pqc = HardwareEfficientNoInput(n_wires=N_WIRES, n_layers=N_LAYERS)

    trainer = BackpropogationTrainer(pqc, q_device, dataset, BATCH_SIZE)
    trained_pqc = trainer.train(plot_loss=plot_results, n_epochs=N_EPOCHS)
    qiskit_circuit = tq2qiskit(q_device, trained_pqc)

    # with open("./circuits/savedQiskitCircuits/" + save_pqc_file_name, 'rb') as handle:
    #     qiskit_circuit = qpy.load(handle)
 
    circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})

    # save trained circuit
    with open("./circuits/savedQiskitCircuits/" + save_pqc_file_name, 'wb') as file:
        qpy.dump(qiskit_circuit, file)

    if plot_results: plot_tq_sim_measurements(q_device, trained_pqc)

# must be in qiskitvenv
def run_on_ibm_quantum(load_pqc_file_name):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.circuit.library import GroverOperator, Diagonal

    from qiskit.quantum_info import Statevector
    from qiskit.visualization import plot_histogram

    from qiskit_ibm_runtime import QiskitRuntimeService, Batch
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    with open("./circuits/savedQiskitCircuits/" + load_pqc_file_name, 'rb') as handle:
        qc = qpy.load(handle)[0]
        qc = qc.reverse_bits()
    
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    ideal_distribution = Statevector.from_instruction(qc).probabilities_dict()
    # Need to add measurements to the circuit

    qc.measure_all()
    pass_manager = generate_preset_pass_manager(3, backend=backend, seed_transpiler=0)
    transpiled_qc = pass_manager.run(qc)
    
    print("running measurements on:", backend.name)
    with Batch(backend=backend):
        sampler = Sampler()
        job = sampler.run(
            [transpiled_qc],
            shots=8000
        )
        result = job.result()

    # run most recent job
    # successful_job = next(j for j in service.jobs() if j.status().name == "DONE")
    # job_id = successful_job.job_id()
    # print(job_id)
    # result = service.job(job_id).result()
    
    binary_prob = [{k: v / res.data.meas.num_shots for k, v in res.data.meas.get_counts().items()} for res in result]
    plot_histogram(
    binary_prob + [ideal_distribution],
    bar_labels=False,
    legend=[
        "optimization_level=3",
        "ideal distribution",
    ],)
    plt.show()

def plot_tq_sim_measurements(q_device, trained_pqc):
    from sklearn.neighbors import KernelDensity
    from utils.helper_functions import evenlySpaceEigenstates 
    from scipy.stats import norm
    from qiskit.visualization import plot_histogram
    # measure with torchquantum simulator
    measurements = trained_pqc(q_device=q_device, measure=True)
    data = measurements[0]

    # create axes
    fig, ax1 = plt.subplots()
    x_values = np.linspace(-1.5, 1.5, 1000)
    ax2 = ax1.twinx()
    
    # plot histogram of measured states
    plot_histogram(data)
    states = [evenlySpaceEigenstates(bitstring, N_WIRES, -1.5, 1.5) for bitstring in data.keys()]
    state_frequencies = list(data.values())
    ax1.bar(states, state_frequencies, label='Histogram of Measurements')

    # plot kernel density estimation of measured states
    expanded_data = []
    for key, freq in data.items():
        value = evenlySpaceEigenstates(key, N_WIRES, -1.5, 1.5)
        expanded_data.extend([value] * freq)
    expanded_data = np.array(expanded_data).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(np.array(expanded_data).reshape(-1, 1))
    kde_values = np.exp(kde.score_samples(x_values.reshape(-1, 1)))
    ax2.plot(x_values, kde_values, color="r", label="KDE of Measurements")


    # plot true distribution
    pdf_values = 0.5*(norm.pdf(x_values, loc=-0.75, scale=0.1) + norm.pdf(x_values, loc=0.75, scale=0.1))
    ax2.plot(x_values, pdf_values, color='g', label="True Distribution")

    plt.show()




