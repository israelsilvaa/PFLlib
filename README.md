# 📘 Tutorial de Instalação e Execução do PFLlib

## 🔧 1. Instale o CUDA (se for usar GPU)

Se você pretende usar **GPU com CUDA**, instale a versão do CUDA **compatível com sua GPU e com a versão do PyTorch**.

🔗 Veja as versões compatíveis aqui:  
https://pytorch.org/get-started/previous-versions/

> 💡 **Se for usar apenas CPU**, pode pular esta etapa.

---

## 🧪 2. Abra o terminal do Anaconda

1. Vá no menu Iniciar
2. Digite **"Anaconda Prompt"**
3. Clique para abrir

---

## 📦 3. Clone o repositório

Se não tiver `git`, instale com:

```bash
    $ conda install git
```

No terminal Anaconda, digite:
```bash
    $ git clone https://github.com/TsingZ0/PFLlib.git
```

```bash
    $ cd PFLlib
```

## 🧬 4. Crie o ambiente Conda

---

💡 Dica: Se usar CPU, troque por `env_cpu.yaml`:

```bash
    $ conda env create -f env_cpu.yaml
```

Depois, ative o ambiente:

```bash
    $ conda activate pfllib
```

```bash
    $ cd system
```

---

## 🚀 5. Rodar o treino

Base: **MNIST**  
| Flag  | Significado              | Valor | Interpretação                                      |
|-------|---------------------------|--------|----------------------------------------------------|
| `-gr` | Global Rounds             | 50     | São as **rodadas de comunicação** (20 rounds)      |
| `-jr` | Join Ratio                | 0.2    | Proporção de **clientes por rodada** (0.2 × 20 = 4)|
| `-nc` | Number of Clients         | 20     | Número total de **clientes disponíveis** no sistema|


```bash
    $ python main.py -data MNIST -m CNN -algo FedAvg -gr 300 -jr 0.2 -nc 100
    $ python main.py -data EMNIST -m CNN -algo FedAvg -gr 300 -jr 0.2 -nc 100
    $ python main.py -data Cifar10 -m CNN -algo FedAvg -gr 300 -jr 0.2 -nc 100
    $ python main.py -data Cifar100 -m CNN -algo FedAvg -gr 300 -jr 0.2 -ncl 100
```

para criar distribuição MNIST, EMNIST, Cifar100 
```bash
    $ python generate_EMNIST.py noniid - dir
    $ python generate_MNIST.py noniid - dir
    $ python generate_Cifar100.py noniid - dir
```

para ver o grafico do ultimo treinamento(esteja em /system)
```bash
    $ python plot_results.py ../results/MNIST_FedAvg_test_0.h5
```

para comparar 1 a 4 treinamentos(esteja em /system)
```bash
    $ python plot_results2.py 0 1 2 4
```



