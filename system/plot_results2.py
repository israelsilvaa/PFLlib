import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

if len(sys.argv) != 4:
    print("Uso: python plot_results2.py <id_1> <id_2> <id_3>")
    sys.exit(1)

# Diretório base dos arquivos
base_dir = "../results"

# Labels fixos na ordem recebida
labels = ["Padrão", "Seleção v2", "Seleção v3"]

# Monta lista de caminhos e mapeia com labels
file_label_map = {}
for i, file_id in enumerate(sys.argv[1:]):
    filename = f"MNIST_FedAvg_test_{file_id}.h5"
    file_path = os.path.join(base_dir, filename)
    file_label_map[file_path] = labels[i]

data = {}

# Carrega dados
for path, label in file_label_map.items():
    if not os.path.exists(path):
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)

    with h5py.File(path, 'r') as f:
        test_acc = f.get('rs_test_acc')
        train_loss = f.get('rs_train_loss')

        if test_acc is None or train_loss is None:
            print(f"Acurácia ou perda de treino não encontrados no arquivo: {path}")
            sys.exit(1)

        data[label] = {
            "acc": test_acc[:],
            "loss": train_loss[:]
        }

# Plots
plt.figure(figsize=(14, 5))

# Acurácia
plt.subplot(1, 2, 1)
for label, d in data.items():
    plt.plot(d["acc"], label=label)
plt.xlabel('Rodada')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia por Rodada')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Perda
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
