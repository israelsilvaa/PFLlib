
    # Teste definitivo         =   acc   perda
    # 0 v1                     = ok
    # 1 v2                     = ok
    # 2 v3 7 3 5 5 - reta 150  = ok
    # 3 v3 fixa so size_score  = ok
    # 4 v3 fixa alfa7 e beta3  = ok
    # 5 v3  = 
    # 6 v3 media full penalizd = 
    # 7 v3 media full penalizd e FULL unicidade = 
    # 8 v3 FULL unicidade      = 
    # 9 v3 37 no db  150 epocs = *
    # 10 v3 37 no db 300 epocs = 
    # 11 v3 37 no db sem acc   = 
    
    # ID  -  versão   -   epocas   - observação                       -    bestAcc  -  mediaAcc   -   loss
    # 14  D  v1           300                                              97.2%       0.9581         0.0901     100 clientes
    # 13  D  v2           300                                              97.3%       0.9581         0.1075
    # 12     v3           300        m7*3 db 3*7 (topk+alt)                95%                        0.1260
    # 15     v3           300        7 3 5 5                               97  %       0.9581         0.0865
    # 16     v3           300        v2 sem acc|só nota do banco           95.1%       0.9581         0.1524 
    # 17     v3           300        7 3 5 5 so top-k    

    # 24     v3           300        topk + aleatorio ESSE FOI BOM, NÇAO APAGAR
    # media (5acc + 5baseDados)/count e a baseDados (0.3 * size_score + 0.7 * uniq_score) 
 
    # 25     v3           300        topk + aleatorio + minist + TOPK INVERTIDO
    # media (3acc + 7baseDados)/count e a baseDados (0.3 * size_score + 0.7 * uniq_score) 
 
    # 26     v3           300        topk + minist + TOPK INVERTIDO      aqui tive 95% MUITO estavel(viciou nos 2,3,13,10) problema, so treinou com esses
    # media (7acc + 3baseDados)/count e a baseDados (0.5 * size_score + 0.5 * uniq_score) 

    # 27     v3           300        topk + aleatorio + minist + TOPK INVERTIDO
    # media (7acc + 3baseDados)/count e a baseDados (0.5 * size_score + 0.5 * uniq_score) 
    
    # 28     v3           300        topk + aleatorio + minist + TOPK INVERTIDO          instavel
    # media (5(acc/count) + 5baseDados) e a baseDados (0.3 * size_score + 0.7 * uniq_score)
    
    # 29     v3           300        topk + aleatorio + minist + TOPK INVERTIDO          sem penalizar
    # media (5(acc/count) + 5baseDados) e a baseDados (0.3 * size_score + 0.7 * uniq_score)
    

    # 29     v4           150        topk + aleatorio + minist + TOPK INVERTIDO
    # media (7(acc/count) + 3baseDados) e a baseDados (0.3 * size_score + 0.7 * uniq_score)
    
    # 30     v4           150        topk + aleatorio + minist
    # media (7(acc/count) + 3baseDados) e a baseDados (0.3 * size_score + 0.7 * uniq_score)
    

    # 33  D  v4           300        topk + aleatorio + minist + TOPK INVERTIDO         TA LEGAL ESSA FOI A MELHOR E EU DEDISTO
    # media (5acc + 5db)/count e a baseDados (0.5 * size_score + 0.5 * uniq_score)



    #          TESTE FINAL
    # ID  -  versão   -   epocas   - observação                       -    bestAcc  -  mediaAcc         -        loss
    # 34  D  v1           300                                              97.3%       0.9581                    0.0901
    # 35  D  v4           300                                              97.4%       0.9581286416085913        0.0987
    # 36  D  v2           300                                              
    
    # 38  D  v4           300        50 clientes                    0.9751 (97.5%)                           
    # 39  D  v1           300        50 clientes                              
    #                
    # 40  D  v1           300        20 clientes                                   
    # 41  D  v4           300        20 clientes                                              
    # 40  D  v1           300        20 clientes  Mnist           Acurácia média = 0.9142, Perda média = 0.2444   X                 
    # 41  D  v4           300        20 clientes  Mnist           Acurácia média = 0.9206, Perda média = 0.2339   X


    #          ROTINA
    #### 42  D  v1           300        20 clientes  Mnist           ESSE AQUI QUE VAI PRO ARITGO
    #### 43  D  v4           300        20 clientes  Mnist           A 

    #### 2  D  v1           300        20 clientes  FashionMnist     Acurácia média = 0.6990, Perda média = 0.9185  X                        
    #### 3  D  v4           300        20 clientes  FashionMnist     Acurácia média = 0.7322, Perda média = 0.7809  X

    # 6  D  v1           300        20 clientes  cifar100         Acurácia média = 0.2313, Perda média = 2.7165  X                              
    # 7  D  v4           300        20 clientes  cifar100         Acurácia média = 0.2333, Perda média = 2.6956  X

    # 4  D  v1           300        20 clientes  cifar10         A                                
    # 5  D  v4           300        20 clientes  cifar10         A  
    # ver o cifar 100 com 100 clinetes?



    # SELECIONA os top-k clientes com melhor desempenho para as próximas rodadas de treino
    # v1.1
    # def select_clients(self):
    #     # Verificar se ao menos um cliente possui test_accuracy > 0.
    #     if any(c.test_accuracy > 0 for c in self.clients):
    #         # Seleciona os clientes com maior acurácia
    #         selected_clients = sorted(
    #             self.clients, 
    #             key=lambda c: c.test_accuracy, 
    #             reverse=True
    #         )[:self.num_join_clients]
    #     else:
    #         # Seleciona aleatoriamente se ninguém foi avaliado ainda
    #         selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))

    #     # Retorna a lista de clientes selecionados (enviesada para os melhores)
    #     return selected_clients
