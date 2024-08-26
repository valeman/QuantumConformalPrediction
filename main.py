from experiments.unsup_learning_exp import train_and_save_model, run_on_ibm_quantum

train_and_save_model("pqc_circuit_deterministic.qpy", plot_results=True, deterministic=True)
# run_on_ibm_quantum("pqc_circuit.qpy")


