# service.py

class ServicoSelecaoClientes:
    VERDE = "\033[92m"
    AMARELO = "\033[93m"
    RESET = "\033[0m"

    def __init__(self, col_width=6):
        self.col_width = col_width

    def exibir_resumo_selecao(self, todos_clientes, top_k, aleatorios):
        ids_colored = []
        acc_colored = []

        for c in todos_clientes:
            if c in top_k:
                cor = self.VERDE
            elif c in aleatorios:
                cor = self.AMARELO
            else:
                cor = self.RESET

            ids_colored.append(f"{cor}{c.id:^{self.col_width}d}{self.RESET}")
            acc_colored.append(f"{cor}{c.test_accuracy:^{self.col_width}.2f}{self.RESET}")

        print("\n\n\n\n\n")
        print("\n--- Seleção para próxima rodada ---")
        print("Clientes (IDs): " + "".join(ids_colored))
        print("Acurácias:      " + "".join(acc_colored))


