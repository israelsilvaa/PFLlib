import json

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

def get_dataset_scores(config_path):

    config = load_config(config_path)
    clients_data = config['Size of samples for labels in clients']

    # pega o maior número total de amostras de um cliente (referência para normalização)
    max_num_samples = max(sum(count for _, count in c) for c in clients_data)

    # soma global de amostras por classe (todos clientes juntos)
    total_per_class = {}
    for client_data in clients_data:
        for label, count in client_data:
            total_per_class[label] = total_per_class.get(label, 0) + count

    # calcula métricas individuais por cliente
    base_metrics = []
    for idx, client_data in enumerate(clients_data):
        # total de amostras do cliente
        num_samples = sum(count for _, count in client_data)
        # score de tamanho: proporcional ao maior cliente
        size_score = num_samples / max_num_samples if max_num_samples > 0 else 0

        if client_data and num_samples > 0:
            # score de "unicidade": quanto das classes desse cliente são exclusivas
            uniqueness_score = sum(
                (count / total_per_class[label]) * (count / num_samples)
                for label, count in client_data
            )
        else:
            uniqueness_score = 0

        base_metrics.append((size_score, uniqueness_score))

    scores = []
    for size_score, uniq_score in base_metrics:   # notal final combinada
        dataset_score = 0.5 * size_score + 0.5 * uniq_score
        scores.append(dataset_score)

    return scores

# Teste rápido local
if __name__ == "__main__":

    config_path = "C:/Users/Israelsilvaa/Documents/GitHub/PFLlib/dataset/MNIST/config.json"
    config_path = "C:/Users/Israelsilvaa/Documents/GitHub/PFLlib/dataset/FashionMNIST/config.json"
    print("== Scores ==")
    print(get_dataset_scores(config_path))

