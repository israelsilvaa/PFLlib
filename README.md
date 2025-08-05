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
    $ python main.py -data MNIST -m CNN -algo FedAvg -gr 30 -jr 0.2 -nc 20
```

para ver o grafico do ultimo treinamento
```bash
    $ python plot_results.py ../results/MNIST_FedAvg_test_0.h5
```


## Sugestões do professor (PFLlib – Aprendizado Federado)

1. **Considerar o tamanho da base de dados de cada cliente**  
   - Exemplo:  
     - Cliente X possui 100 imagens → peso = `100 / 100 = 1.0`  
     - Cliente Y possui 50 imagens → peso = `50 / 100 = 0.5`  
   - Esse peso deve ser levado em conta na **seleção dos clientes** para teste, combinando:
     - `client.acuracia`
     - `client.importanciaDaBaseDeDados` (proporcional ao tamanho da base local)

2. **Suavizar o gráfico de desempenho**
   - Estratégias sugeridas:
     - Aumentar o número de rodadas de treinamento
     - Melhorar o aprendizado (ajustar taxa de aprendizado, regularização, etc.)



## Resumo: Melhoria na Seleção de Clientes com Base de Dados no Aprendizado Federado

### Objetivo
Melhorar a seleção dos clientes no aprendizado federado, levando em consideração **dois fatores** principais:
1. **Acurácia de Teste** (`c.test_accuracy`)
2. **Importância da Base de Dados Local** (`c.data_importance`)

### O que é feito:
- **Acurácia Ajustada**: A acurácia de cada cliente é dividida pelo número de vezes que ele foi selecionado, para evitar que clientes que já participaram muitas vezes sejam escolhidos novamente.
  
- **Importância da Base de Dados**: A importância da base de dados de um cliente é calculada como a fração de seus dados em relação ao total de dados de todos os clientes.

### Como calcular:
1. **Acurácia ajustada**:
   ```python
       acuracia_ajustada = c.test_accuracy / (1 + c.selection_count)
   ```
2. Importância da base de dados:
    ```python
    c.data_importance = c.num_samples / total_num_samples
    ```
   
Média ponderada (usando pesos iguais de 50%):
    ```python
    score = (0.5 * acuracia_ajustada) + (0.5 * c.data_importance)
    ```


