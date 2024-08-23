import torchquantum as tq
from torchquantum.plugin import tq2qiskit
from torch.utils.data import TensorDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

from training.unsup_density_trainer import BackpropogationTrainer
from circuits import HardwareEfficientNoInput
from utils import combinedNormals
from utils.helper_functions import toClosestEigenstate, evenlySpaceEigenstates



N_WIRES = 5
N_LAYERS = 2
N_EPOCHS = 70
BATCH_SIZE = 20
N_TRAINING_SAMPLES = 50

def unsup_learning_exp():
    # create training data
    dist = combinedNormals(-0.75, 0.1, 0.75, 0.1)
    training_samples = toClosestEigenstate(torch.from_numpy(dist.rvs(size=N_TRAINING_SAMPLES)), N_WIRES, -1.5, 1.5)
    dataset = TensorDataset(training_samples)

    # create and train pqc
    q_device = tq.QuantumDevice(n_wires=N_WIRES)
    pqc = HardwareEfficientNoInput(n_wires=N_WIRES, n_layers=N_LAYERS)
    trainer = BackpropogationTrainer(pqc, q_device, dataset, BATCH_SIZE)
    trained_pqc = trainer.train(plot_loss=True, n_epochs=N_EPOCHS)

    # convert circuit to qiskit and draw
    qiskit_circuit = tq2qiskit(q_device, trained_pqc)
    circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})
    plt.show()

    # measure with torchquantum simulator
    measurements = trained_pqc(q_device=q_device, measure=True)
    data = measurements[0]

    # create axes
    fig, ax1 = plt.subplots()
    x_values = np.linspace(-1.5, 1.5, 1000)
    ax2 = ax1.twinx()
    
    # plot histogram of measured states
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