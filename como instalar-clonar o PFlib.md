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
Rodadas: **20**  
Clientes totais: **20**  
Clientes por rodada: **20%**

```bash
    $ python main.py -data MNIST -m CNN -algo FedAvg -gr 20 -jr 0.2 -nc 20
```

para ver o grafico do ultimo treinamento
```bash
    $ python plot_results.py ../results/MNIST_FedAvg_test_0.h5
```
