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

para ver o grafico do ultimo treinamento(esteja em /system)
```bash
    $ python plot_results.py ../results/MNIST_FedAvg_test_0.h5
```

para comparar 3 treinamentos(esteja em /system)
```bash
    $ python plot_results2.py ../results/MNIST_FedAvg_test_0.h5 ../results/MNIST_FedAvg_test_1.h5 ../results/MNIST_FedAvg_test_2.h5
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



info aqui::::
import json

# Caminho do arquivo JSON
config_path = "/home/kurumi/Downloads/PFLlib-israel-trab2/dataset/MNIST/config.json"

# Função para carregar o arquivo JSON
def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

# Função para calcular a pontuação de cada cliente com base na importância dos dados
def calculate_data_importance(config):
    # Número total de amostras de todos os clientes
    total_num_samples = sum(sum(count for _, count in client_data) for client_data in config['Size of samples for labels in clients'])
    
    print("\n*** Cálculo das Pontuações de Importância dos Dados dos Clientes ***\n")
    
    # Lista para armazenar as pontuações
    scores = []
    
    # Para cada cliente, calcular a importância da base de dados
    for idx, client_data in enumerate(config['Size of samples for labels in clients']):
        # Número de amostras para esse cliente
        num_samples = sum(count for _, count in client_data)
        
        # Cálculo da importância da base de dados
        data_importance = num_samples / total_num_samples
        
        # Armazenando o resultado
        scores.append({
            "cliente": idx,
            "num_amostras": num_samples,
            "importancia_dados": data_importance
        })
        
        # Exibindo os resultados para cada cliente
        print(f"Cliente {idx}:")
        print(f"  Número de amostras: {num_samples}")
        print(f"  Importância da base de dados: {data_importance:.4f}")
        print("-" * 40)
    
    # Exibindo as pontuações finais ordenadas por importância
    print("\n*** Ranking de Clientes por Importância dos Dados ***")
    sorted_scores = sorted(scores, key=lambda x: x['importancia_dados'], reverse=True)
    
    for rank, score in enumerate(sorted_scores, 1):
        print(f"{rank}. Cliente {score['cliente']} - Importância dos Dados: {score['importancia_dados']:.4f}")

    print("\n*** Fim do Cálculo das Pontuações ***")

# Carregar o arquivo JSON
config = load_config(config_path)

# Calcular a importância dos dados dos clientes
calculate_data_importance(config)
