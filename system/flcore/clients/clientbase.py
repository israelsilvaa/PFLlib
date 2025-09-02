import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        
        # CORREÇÃO 1: Variáveis para controlar acurácia e penalização
        self.raw_test_accuracy = 0.0001  # Acurácia bruta (sem penalização)
        self.local_test_accuracy = 0.0  # Acurácia local após treinamento
        self.selection_count = 1  # Contador de seleções
        self.was_trained_this_round = False  # Flag para indicar se foi treinado nesta rodada
        self.last_training_round = -1  # Última rodada em que foi treinado

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    # CORREÇÃO 2: Propriedade que calcula acurácia penalizada automaticamente
    @property
    def penalized_accuracy(self):
        """
        Calcula a acurácia penalizada baseada no número de seleções.
        Fórmula: acurácia_bruta / (1 + selection_count)
        """
        if self.selection_count == 0:
            return self.raw_test_accuracy
        return self.raw_test_accuracy 
    
    @property 
    def test_accuracy(self):
        """
        Propriedade que retorna a acurácia penalizada para compatibilidade
        """
        return self.penalized_accuracy
    

    # CORREÇÃO 3: Método para calcular acurácia local após treinamento
    def calculate_local_accuracy(self):
        """Calcula a acurácia local do cliente após o treinamento"""
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        # Atualiza a acurácia bruta (sem penalização)
        self.local_test_accuracy = test_acc / test_num if test_num > 0 else 0.0
        self.raw_test_accuracy = self.local_test_accuracy  # Atualiza a acurácia bruta
        
        return self.local_test_accuracy

    def test_metrics(self):
        """Método original mantido para compatibilidade com o servidor"""
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        # CORREÇÃO 4: Atualiza raw_test_accuracy apenas se foi treinado nesta rodada
        current_accuracy = test_acc / test_num if test_num > 0 else 0.0
        
        # Se foi treinado nesta rodada, usa a acurácia local calculada
        if self.was_trained_this_round:
            self.raw_test_accuracy = self.local_test_accuracy
        # Caso contrário, mantém a acurácia bruta anterior (não atualiza)
        
        # A propriedade test_accuracy retornará automaticamente a versão penalizada
        return test_acc, test_num, auc

    def train_metrics(self):
        """Método para calcular métricas de treinamento"""
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    # CORREÇÃO 5: Marcar quando o cliente foi treinado (com incremento automático)
    def mark_as_trained(self, round_num):
        """Marca o cliente como treinado nesta rodada e incrementa contador de seleções"""
        self.was_trained_this_round = True
        self.last_training_round = round_num
        self.selection_count += 1  # Incrementa automaticamente o contador

    def reset_training_flag(self):
        """Reseta a flag de treinamento para a próxima rodada"""
        self.was_trained_this_round = False

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
