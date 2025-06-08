import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

file_path = sys.argv[1]

with h5py.File(file_path, 'r') as f:
    test_acc = f.get('rs_test_acc')
    train_loss = f.get('rs_train_loss')

    if test_acc is None or train_loss is None:
        print("Acurácia ou perda de treino não encontrados no arquivo.")
        sys.exit(1)

    test_acc = test_acc[:]
    train_loss = train_loss[:]

# Identifica a maior acurácia e sua rodada
max_acc = np.max(test_acc)
max_round = np.argmax(test_acc)

plt.figure(figsize=(12, 5))

# Plot da Acurácia
plt.subplot(1, 2, 1)
plt.plot(test_acc, label='Acurácia', color='green')
plt.axhline(y=max_acc, color='blue', linestyle='--', label='Melhor Acurácia')
plt.annotate(f'{max_acc:.2%} (Rodada {max_round})',
             xy=(max_round, max_acc),
             xytext=(max_round, max_acc + 0.02),
             arrowprops=dict(facecolor='blue', arrowstyle='->'),
             fontsize=9)

plt.xlabel('Rodada')
plt.ylabel('Acurácia')
plt.title('Acurácia por Rodada')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.grid(True, linestyle='--', alpha=0.6)  # ← grade no fundo
plt.legend()

# Plot da Perda
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Perda', color='red')
plt.xlabel('Rodada')
plt.ylabel('Perda')
plt.title('Perda por Rodada')
plt.grid(True, linestyle='--', alpha=0.6)  # ← grade no fundo
plt.legend()

plt.tight_layout()
plt.show()
