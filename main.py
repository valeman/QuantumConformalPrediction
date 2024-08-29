from experiments.regression_exp import train_and_save_model
import torchquantum as tq
import matplotlib.pyplot as plt
import torch
from circuits.hardware_eff_input import HardwareEfficientInput


train_and_save_model("tester.qpy", "IP")

# Example state tensor and targets tensor

