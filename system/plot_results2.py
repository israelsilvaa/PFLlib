import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import numpy as np

# -------- Configuração de suavização --------
smooth = True       # <<<<<< chave: coloque False se não quiser suavizar
window_size = 1    # tamanho da janela da média móvel

def moving_average(x, w):
    """Aplica média móvel em um array"""
    return np.convolve(x, np.ones(w), 'valid') / w

# Checa número de argumentos
if len(sys.argv) < 2 or len(sys.argv) > 5:
    print("Uso: python plot_results2.py <id_1> [id_2] [id_3] [id_4]")
    sys.exit(1)

# Diretório base dos arquivos
base_dir = "../results"

# Labels fixos
labels = ["Padrão", "v2", "v3", "v4"]

# Monta lista de caminhos e mapeia com labels (até o número de argumentos passados)
file_label_map = {}
for i, file_id in enumerate(sys.argv[1:]):
    # Mantive todos os datasets como no seu código
    filename = f"Cifar10_FedAvg_test_{file_id}.h5"
    filename = f"EMNIST_FedAvg_test_{file_id}.h5"
    filename = f"FashionMNIST_FedAvg_test_{file_id}.h5"
    filename = f"Cifar100_FedAvg_test_{file_id}.h5"
    filename = f"Cifar10_FedAvg_test_{file_id}.h5"
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

# -------- Cálculo das médias finais --------
n_tail = 50
print(f"\n=== Médias das últimas {n_tail} rodadas ===")
for label, d in data.items():
    acc_mean = np.mean(d["acc"][-n_tail:])
    loss_mean = np.mean(d["loss"][-n_tail:])
    print(f"{label:8s} -> Acurácia média = {acc_mean:.4f}, Perda média = {loss_mean:.4f}")

# -------- Análises adicionais --------
print("\n=== Análises adicionais ===")

# 1. Quem convergiu mais nas primeiras 50 rodadas
early_rounds = 50
for label, d in data.items():
    acc_mean_early = np.mean(d["acc"][:early_rounds])
    print(f"{label:8s} -> Média de acurácia nas primeiras {early_rounds} rodadas = {acc_mean_early:.4f}")
print("\n")

# 2. Quem é mais estável (desvio padrão nas últimas N rodadas)
for label, d in data.items():
    acc_std = np.std(d["acc"][-n_tail:])
    print(f"{label:8s} -> Desvio padrão (estabilidade) = {acc_std:.4f}")



# -------- Plots --------
plt.figure(figsize=(14, 5))

# Acurácia
plt.subplot(1, 2, 1)
for label, d in data.items():
    acc = d["acc"]
    if smooth:
        acc = moving_average(acc, window_size)
    plt.plot(acc, label=label)
plt.xlabel('Rodada')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia por Rodada')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Perda
plt.subplot(1, 2, 2)
for label, d in data.items():
    loss = d["loss"]
    if smooth:
        loss = moving_average(loss, window_size)
    plt.plot(loss, label=label)
plt.xlabel('Rodada')
plt.ylabel('Perda')
plt.title('Comparação de Perda por Rodada')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
