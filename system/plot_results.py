import sys
import h5py
import matplotlib.pyplot as plt

file_path = sys.argv[1]

with h5py.File(file_path, 'r') as f:
    print("Datasets disponíveis no arquivo:")
    for key in f.keys():
        print(key)

    test_acc = f.get('rs_test_acc')
    train_loss = f.get('rs_train_loss')

    if test_acc is None or train_loss is None:
        print("Acurácia ou perda de treino não encontrados no arquivo.")
        sys.exit(1)

    test_acc = test_acc[:]
    train_loss = train_loss[:]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(test_acc, label='Test Accuracy')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Test Accuracy por Rodada')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_loss, label='Train Loss', color='orange')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('Train Loss por Rodada')
plt.legend()

plt.tight_layout()
plt.show()
