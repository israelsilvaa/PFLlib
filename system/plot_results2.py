import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

if len(sys.argv) != 4:
    print("Uso: python plot_results.py <arquivo_0.h5> <arquivo_1.h5> <arquivo_2.h5>")
    sys.exit(1)

file_paths = sys.argv[1:]

# Mapeia nomes de arquivos para rótulos amigáveis
file_label_map = {
    "MNIST_FedAvg_test_0": "Padrão (Non-IID)",
    "MNIST_FedAvg_test_1": "Seleção v1 (Non-IID)",
    "MNIST_FedAvg_test_2": "Seleção v2 (Non-IID)",
    "MNIST_FedAvg_test_3": "Padrão (IID)",
    "MNIST_FedAvg_test_4": "Seleção v1 (IID)",
    "MNIST_FedAvg_test_5": "Seleção v2 (IID)",
}

data = {}

for path in file_paths:
    with h5py.File(path, 'r') as f:
        test_acc = f.get('rs_test_acc')
        train_loss = f.get('rs_train_loss')

        if test_acc is None or train_loss is None:
            print(f"Acurácia ou perda de treino não encontrados no arquivo: {path}")
            sys.exit(1)

        base = os.path.basename(path).replace(".h5", "")
        label = file_label_map.get(base, base)
        data[label] = {
            "acc": test_acc[:],
            "loss": train_loss[:]
        }

plt.figure(figsize=(14, 5))

# Subplot da Acurácia
plt.subplot(1, 2, 1)
for label, d in data.items():
    plt.plot(d["acc"], label=label)
plt.xlabel('Rodada')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia por Rodada')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Subplot da Perda
plt.subplot(1, 2, 2)
for label, d in data.items():
    plt.plot(d["loss"], label=label)
plt.xlabel('Rodada')
plt.ylabel('Perda')
plt.title('Comparação de Perda por Rodada')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
