from qiskit import qpy
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer


N_WIRES = 5
N_LAYERS = 2
N_EPOCHS = 70
BATCH_SIZE = 20
N_TRAINING_SAMPLES = 50

def train_and_save_model(save_pqc_file_name, model_type, plot_results=True):
    import torchquantum as tq
    from torchquantum.plugin import tq2qiskit 
    from torch.utils.data import TensorDataset
    import torch
    import torch.nn as nn


        
    from circuits import HardwareEfficientNoInput
    from utils.combined_normals import combinedNormals
    from utils.helper_functions import toClosestEigenstate, evenlySpaceEigenstates

    # create training data
    dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
    training_samples = toClosestEigenstate(torch.from_numpy(dist.rvs(size=N_TRAINING_SAMPLES)), N_WIRES, -1.5, 1.5)
    dataset = TensorDataset(training_samples)

    q_device = tq.QuantumDevice(n_wires=N_WIRES)

    # create and train pqc
    pqc = HardwareEfficientNoInput(n_wires=N_WIRES, n_layers=N_LAYERS)


    if model_type == "IP":
        from training.implicit_probabilistic_trainer import BackpropogationTrainer
        trainer = BackpropogationTrainer(pqc, q_device, dataset, BATCH_SIZE)
    elif model_type == "D":
        from training.deterministic_trainer import BackpropogationTrainer
        eigenvalues = evenlySpaceEigenstates(torch.arange(start=0, end= 2**N_WIRES, step=1), N_WIRES, -1.5, 1.5)
        trainer = BackpropogationTrainer(pqc, q_device, dataset, BATCH_SIZE, eigenvalues)
    else:
        raise ValueError("D or IP for model type")

    trained_pqc = trainer.train(plot_loss=plot_results, n_epochs=N_EPOCHS)
    qiskit_circuit = tq2qiskit(q_device, trained_pqc)
    # with open("./circuits/savedQiskitCircuits/" + save_pqc_file_name, 'rb') as handle:
    #     qiskit_circuit = qpy.load(handle)
 
    circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})

    # save trained circuit
    with open("./circuits/savedQiskitCircuits/" + save_pqc_file_name, 'wb') as file:
        qpy.dump(qiskit_circuit, file)

    if plot_results: plot_tq_sim_measurements(q_device, trained_pqc)

# MUST BE IN QTVENV
def plot_tq_sim_measurements(q_device, trained_pqc):
    from sklearn.neighbors import KernelDensity
    from utils.helper_functions import evenlySpaceEigenstates 
    from scipy.stats import norm
    from qiskit.visualization import plot_histogram
    import torch

    # measure with torchquantum simulator
    measurements = trained_pqc.sample_from_model(1000)
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


    eigenvalues = evenlySpaceEigenstates(torch.arange(start=0, end= 2**N_WIRES, step=1), N_WIRES, -1.5, 1.5)
    exp_val = trained_pqc.calculate_expected_value(eigenvalues).item()
    ax2.axvline(x=exp_val, color='r', linestyle='--', label=f'expected_value = {exp_val}')


    plt.show()




