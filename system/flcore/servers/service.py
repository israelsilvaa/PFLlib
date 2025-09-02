
class ServicoSelecaoClientes:
    """Serviço para exibir resumo detalhado da seleção de clientes"""
    
    VERDE = "\033[92m"      # Top-k selecionados
    AMARELO = "\033[93m"    # Aleatórios selecionados
    LARANJA = "\033[94m"       # Não selecionados
    RESET = "\033[0m"

    def __init__(self, col_width=8):
        self.col_width = col_width

    def exibir_resumo_selecao(self, todos_clientes, top_k, aleatorios, medias_ponderadas=None):
        """
        Exibe resumo completo da seleção de clientes
        
        Args:
            todos_clientes: Lista de todos os clientes
            top_k: Lista dos clientes selecionados pelo critério top-k(melhor acc)
            aleatorios: Lista dos clientes selecionados aleatoriamente
            medias_ponderadas: Dicionário {cliente_id: media_ponderada}
        """
        ids_colored = []
        acc_bruta_colored = []
        acc_penalizada_colored = []
        sel_colored = []
        medias_colored = []

        for c in todos_clientes:
            # Determina a cor baseada no tipo de seleção
            if c in top_k:
                cor = self.VERDE
                tipo = "T"  # Top-k
            elif c in aleatorios:
                cor = self.AMARELO
                tipo = "A"  # Aleatório
            else:
                cor = self.LARANJA
                tipo = "-"  # Não selecionado

            # Formata as informações com cores
            ids_colored.append(f"{cor}{c.id:^{self.col_width}d}{self.RESET}")
            acc_bruta_colored.append(f"{cor}{c.raw_test_accuracy:^{self.col_width}.3f}{self.RESET}")
            acc_penalizada_colored.append(f"{cor}{c.penalized_accuracy:^{self.col_width}.3f}{self.RESET}")
            sel_colored.append(f"{cor}{c.selection_count:^{self.col_width}d}{self.RESET}")
            
            # Adiciona média ponderada se fornecida
            if medias_ponderadas and c.id in medias_ponderadas:
                media = medias_ponderadas[c.id]
                medias_colored.append(f"{cor}{media:^{self.col_width}.3f}{self.RESET}")
            else:
                medias_colored.append(f"{cor}{'N/A':^{self.col_width}}{self.RESET}")

        # Exibe o cabeçalho
        print(f"\n{'-'*15} 📊 RESUMO DA SELEÇÃO DE CLIENTES {'-'*15}")
        
        # Exibe as informações tabulares
        print("IDs:            " + "".join(ids_colored))
        print("Acc Bruta:      " + "".join(acc_bruta_colored))
        print("Acc Penalizada: " + "".join(acc_penalizada_colored))
        print("Seleções:     "+"".join(sel_colored))
        
        if medias_ponderadas:
            print("Média Final:    " + "".join(medias_colored))

