from qiskit import qpy
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer


N_WIRES = 5
N_LAYERS = 5
N_EPOCHS = 50
BATCH_SIZE = 50
N_TRAINING_SAMPLES = 100

def train_and_save_model(save_pqc_file_name, model_type, plot_results=True):
    import torchquantum as tq
    from torchquantum.plugin import tq2qiskit 
    from torch.utils.data import TensorDataset
    import torch
    import torch.nn as nn
    from circuits.hardware_eff_input import HardwareEfficientInput
    from utils.sinusoidal_data import SinusoidalData
    from utils.helper_functions import toClosestEigenstate, evenlySpaceEigenstates

    # create training data
    dist = SinusoidalData()
    train_x, train_y = dist.get_data_points(n_points=N_TRAINING_SAMPLES)
    train_y = toClosestEigenstate(train_y, N_WIRES, -2, 2)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))

    q_device = tq.QuantumDevice(n_wires=N_WIRES, bsz=BATCH_SIZE)

    # create and train pqc
    pqc = HardwareEfficientInput(n_wires=N_WIRES, n_layers=N_LAYERS, bsz=BATCH_SIZE)


    from training.regression_trainer import BackpropogationTrainer
    trainer = BackpropogationTrainer(pqc, q_device, dataset, BATCH_SIZE)


    trained_pqc = trainer.train(plot_loss=plot_results, n_epochs=N_EPOCHS)

    plot_tq_sim_measurements(q_device, trained_pqc)

# MUST BE IN QTVENV
def plot_tq_sim_measurements(q_device, trained_pqc):
    from utils.helper_functions import evenlySpaceEigenstates 
    from scipy.stats import norm
    from qiskit.visualization import plot_histogram
    import torch

    # measure with torchquantum simulator
    x_points = np.linspace(-10, 10, 200)
    samples = trained_pqc.sample_from_model(torch.from_numpy(x_points))

    # create axes
    fig, ax1 = plt.subplots()
    continuous_x = np.linspace(-10, 10, 1000)
    
    # plot histogram of measured states
    ax1.scatter(x_points, samples, label='Scatter of Measurements')

    # plot true curve
    continuous_y = 0.5*np.sin(0.8*continuous_x) + 0.05*continuous_x
    ax1.plot(continuous_x, continuous_y, color='g', label="True Distribution")
    ax1.plot(continuous_x, -continuous_y, color='g', label="True Distribution")

    plt.show()
