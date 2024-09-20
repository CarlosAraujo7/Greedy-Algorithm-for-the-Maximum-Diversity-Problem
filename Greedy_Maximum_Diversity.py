import os
import time
import csv
import numpy as np

def calcular_diversidade_maxima(matriz_distancias, k, p):
    n = len(matriz_distancias)
    melhor_diversidade = 0
    melhor_selecao = []

    # Calcular a soma das distâncias de cada vértice para todos os outros
    soma_distancias = np.sum(matriz_distancias, axis=1)
    
    # Selecionar os 'p' vértices com maior soma de distâncias (neste caso, todos)
    vertices_iniciais = np.argsort(-soma_distancias)[:p]
    
    for i in vertices_iniciais:
        selecao_atual = [i]
        diversidade_atual = 0
        candidatos = np.setdiff1d(np.arange(n), selecao_atual)
        
        ganhos = matriz_distancias[candidatos][:, i]
        
        while len(selecao_atual) < k:
            melhor_idx = np.argmax(ganhos)
            melhor_adicao = candidatos[melhor_idx]
            melhor_ganho = ganhos[melhor_idx]
            
            selecao_atual.append(melhor_adicao)
            diversidade_atual += melhor_ganho
            candidatos = np.delete(candidatos, melhor_idx)
            ganhos = np.delete(ganhos, melhor_idx)
            
            if len(candidatos) > 0:
                ganhos += matriz_distancias[candidatos][:, melhor_adicao]
        
        if diversidade_atual > melhor_diversidade:
            melhor_diversidade = diversidade_atual
            melhor_selecao = selecao_atual

    return melhor_diversidade, melhor_selecao


def ler_instancia_txt(caminho_arquivo):
    with open(caminho_arquivo, 'r') as file:
        n, k = map(int, file.readline().strip().split())
        
        matriz_distancias = np.zeros((n, n))
        
        for linha in file:
            vertice1, vertice2, peso = linha.strip().split()
            vertice1 = int(vertice1)
            vertice2 = int(vertice2)
            peso = float(peso)
            matriz_distancias[vertice1][vertice2] = peso
            matriz_distancias[vertice2][vertice1] = peso
                
    return matriz_distancias, k

def solver_diversidade(instancia_file):
    matriz_distancias, k = ler_instancia_txt(instancia_file)
    n = len(matriz_distancias)  # Obter o número de vértices
    
    start_time = time.time()
    # Chamar a função com o parâmetro 'p'
    valor_diversidade_maxima, vertices_selecionados = calcular_diversidade_maxima(matriz_distancias, k, p=n)
    tempo_gasto = time.time() - start_time
    
    # Como o valor ótimo é desconhecido, assumimos que o resultado obtido é o melhor possível
    valor_otimo = valor_diversidade_maxima
    gap = (valor_otimo - valor_diversidade_maxima) / valor_otimo if valor_otimo != 0 else 0
    
    return valor_diversidade_maxima, gap, vertices_selecionados, tempo_gasto


def processar_pasta_instancias(pasta_instancias, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Instância', 'Valor da Diversidade Máxima', 'GAP', 'Vértices Selecionados', 'Tempo Gasto (s)'])
        
        for arquivo in os.listdir(pasta_instancias):
            if arquivo.endswith(".txt"):
                caminho_arquivo = os.path.join(pasta_instancias, arquivo)
                try:
                    valor_diversidade_maxima, gap, vertices_selecionados, tempo_gasto = solver_diversidade(caminho_arquivo)
                    writer.writerow([arquivo, valor_diversidade_maxima, gap, vertices_selecionados, tempo_gasto])
                except Exception as e:
                    print(f"Erro ao processar {arquivo}: {e}")

# Exemplo de uso
pasta_instancias = '../Instances'  # Pasta onde estão os arquivos TXT
output_csv = '../Great values/great_values_2.csv'

processar_pasta_instancias(pasta_instancias, output_csv)
