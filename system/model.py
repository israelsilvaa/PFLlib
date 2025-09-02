import torch
from flcore.trainmodel.models import BaseHeadSplit
from torch.serialization import safe_globals

checkpoint_path = r"C:\Users\Israelsilvaa\Documents\GitHub\PFLlib\system\models\MNIST\FedAvg_server.pt"

with safe_globals([BaseHeadSplit]):
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Pegar os pesos da primeira camada conv
conv1_weights = model.state_dict()['base.conv1.0.weight']

print(conv1_weights)  # imprime todos os valores
print(conv1_weights.shape)  # imprime o shape para referência
