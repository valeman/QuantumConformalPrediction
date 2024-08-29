import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse

import torchquantum as tq

from torch.optim.lr_scheduler import CosineAnnealingLR

import random
import numpy as np

# data is cos(theta)|000> + e^(j * phi)sin(theta) |111>

from torchpack.datasets.dataset import Dataset
from torchquantum.plugin import (
    tq2qiskit_initialize,
    tq2qiskit,
    tq2qiskit_measurement,
    qiskit_assemble_circs,
)


def gen_data(L, N):
    omega_0 = np.zeros([2**L], dtype="complex_")
    omega_0[0] = 1 + 0j

    omega_1 = np.zeros([2**L], dtype="complex_")
    omega_1[-1] = 1 + 0j

    states = np.zeros([N, 2**L], dtype="complex_")

    thetas = 2 * np.pi * np.random.rand(N)
    phis = 2 * np.pi * np.random.rand(N)

    for i in range(N):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )

    X = np.sin(2 * thetas) * np.cos(phis)

    return states, X


class RegressionDataset:
    def __init__(self, split, n_samples, n_wires):
        self.split = split
        self.n_samples = n_samples
        self.n_wires = n_wires

        self.states, self.Xlabel = gen_data(self.n_wires, self.n_samples)

    def __getitem__(self, index: int):
        instance = {"states": self.states[index], "Xlabel": self.Xlabel[index]}
        return instance

    def __len__(self) -> int:
        return self.n_samples


class Regression(Dataset):
    def __init__(self, n_train, n_valid, n_wires):
        n_samples_dict = {"train": n_train, "valid": n_valid}
        super().__init__(
            {
                split: RegressionDataset(
                    split=split, n_samples=n_samples_dict[split], n_wires=n_wires
                )
                for split in ["train", "valid"]
            }
        )


class QModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()
            # inside one block, we have one u3 layer one each qubit and one layer
            # cu3 layer with ring connection
            self.n_wires = n_wires
            self.n_blocks = n_blocks
            self.rx_layers = tq.QuantumModuleList()
            self.ry_layers = tq.QuantumModuleList()
            self.rz_layers = tq.QuantumModuleList()
            self.cnot_layers = tq.QuantumModuleList()

            for _ in range(n_blocks):
                self.rx_layers.append(
                    tq.Op1QAllLayer(
                        op=tq.RX,
                        n_wires=n_wires,
                        has_params=True,
                        trainable=True,
                    )
                )
                self.ry_layers.append(
                    tq.Op1QAllLayer(
                        op=tq.RY,
                        n_wires=n_wires,
                        has_params=True,
                        trainable=True,
                    )
                )
                self.rz_layers.append(
                    tq.Op1QAllLayer(
                        op=tq.RZ,
                        n_wires=n_wires,
                        has_params=True,
                        trainable=True,
                    )
                )
                self.cnot_layers.append(
                    tq.Op2QAllLayer(
                        op=tq.CNOT,
                        n_wires=n_wires,
                        has_params=False,
                        trainable=False,
                        circular=True,
                    )
                )

        def forward(self, q_device: tq.QuantumDevice):
            for k in range(self.n_blocks):
                self.rx_layers[k](q_device)
                self.ry_layers[k](q_device)
                self.rz_layers[k](q_device)
                self.cnot_layers[k](q_device)

    def __init__(self, n_wires, n_blocks):
        super().__init__()
        self.q_layer = self.QLayer(n_wires=n_wires, n_blocks=n_blocks)
        self.encoder = tq.StateEncoder()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, input_states, use_qiskit=False):
        self.q_device = q_device
        # firstly set the q_device states
        # q_device.set_states(input_states)
        devi = input_states.device
        if use_qiskit:
            encoder_circs = tq2qiskit_initialize(
                q_device, input_states.detach().cpu().numpy()
            )
            q_layer_circ = tq2qiskit(q_device, self.q_layer)
            measurement_circ = tq2qiskit_measurement(q_device, self.measure)
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )
            res = self.qiskit_processor.process_ready_circs(
                self.q_device, assembled_circs
            ).to(devi)
        else:
            self.encoder(q_device, input_states)
            self.q_layer(q_device)
            res = self.measure(q_device)

        return res


def train(dataflow, q_device, model, device, optimizer, qiskit=False):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["states"].to(device).to(torch.complex64)
        targets = feed_dict["Xlabel"].to(device).to(torch.float)

        outputs = model(q_device, inputs, qiskit)

        loss = F.mse_loss(outputs[:, 1], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}")


def valid_test(dataflow, q_device, split, model, device, qiskit):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["states"].to(device).to(torch.complex64)
            targets = feed_dict["Xlabel"].to(device).to(torch.float)

            outputs = model(q_device, inputs, qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    loss = F.mse_loss(output_all[:, 1], target_all)

    print(f"{split} set loss: {loss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument(
        "--bsz", type=int, default=32, help="batch size for training and validation"
    )
    parser.add_argument("--n_wires", type=int, default=3, help="number of qubits")
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=2,
        help="number of blocks, each contain one layer of "
        "U3 gates and one layer of CU3 with "
        "ring connections",
    )
    parser.add_argument(
        "--n_train", type=int, default=100, help="number of training samples"
    )
    parser.add_argument(
        "--n_valid", type=int, default=100, help="number of validation samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of training epochs"
    )

    args = parser.parse_args()

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = Regression(
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_wires=args.n_wires,
    )

    dataflow = dict()

    for split in dataset:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(dataset[split])
        else:
            sampler = torch.utils.data.SequentialSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.bsz,
            sampler=sampler,
            num_workers=1,
            pin_memory=True,
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = QModel(n_wires=args.n_wires, n_blocks=args.n_blocks).to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    q_device = tq.QuantumDevice(n_wires=args.n_wires)
    q_device.reset_states(bsz=args.bsz)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}, RL: {optimizer.param_groups[0]['lr']}")
        train(dataflow, q_device, model, device, optimizer)

        # valid
        valid_test(dataflow, q_device, "valid", model, device, False)
        scheduler.step()

    try:
        from qiskit import IBMQ
        from torchquantum.plugin import QiskitProcessor

        print(f"\nTest with Qiskit Simulator")
        processor_simulation = QiskitProcessor(use_real_qc=False)
        model.set_qiskit_processor(processor_simulation)
        valid_test(dataflow, q_device, "test", model, device, qiskit=True)

        # final valid
        valid_test(dataflow, q_device, "valid", model, device, True)
    except:
        pass

    qiskit_circuit = tq2qiskit(q_device, model)
    from qiskit.visualization import circuit_drawer
    import matplotlib.pyplot as plt 
    circuit_drawer(qiskit_circuit, output='mpl', style={'name': 'bw'})
    plt.show()

if __name__ == "__main__":
    main()