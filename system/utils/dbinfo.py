import json

def load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

def get_dataset_scores(config_path, alpha=0.5, beta=0.5):
    """
    Lê o config.json e retorna uma lista com dataset_score de cada cliente.
    """
    config = load_config(config_path)
    clients_data = config['Size of samples for labels in clients']

    max_num_samples = max(sum(count for _, count in c) for c in clients_data)

    total_per_class = {}
    for client_data in clients_data:
        for label, count in client_data:
            total_per_class[label] = total_per_class.get(label, 0) + count

    scores = []
    for idx, client_data in enumerate(clients_data):
        num_samples = sum(count for _, count in client_data)
        size_score = num_samples / max_num_samples

        if client_data:
            uniqueness_values = [count / total_per_class[label] for label, count in client_data]
            uniqueness_score = sum(uniqueness_values) / len(uniqueness_values)
        else:
            uniqueness_score = 0

        dataset_score = alpha * size_score + beta * uniqueness_score
        scores.append(dataset_score)

    return scores

# Teste rápido local
if __name__ == "__main__":
    config_path = "C:/Users/Israelsilvaa/Documents/GitHub/PFLlib/dataset/MNIST/config.json"
    scores = get_dataset_scores(config_path)
    for i, s in enumerate(scores):
        print(f"Cliente {i} - Dataset Score: {s:.4f}")
