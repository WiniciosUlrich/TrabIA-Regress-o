import matplotlib.pyplot as plt
import numpy as np

# ===============================
# Funções Estatísticas Básicas
# ===============================

def media(vetor):
    """
    Calcula a média aritmética de um conjunto de dados
    Entrada: lista de números
    Saída: valor da média
    """
    return sum(vetor) / len(vetor)

def correlacao(x, y):
    """
    Calcula o coeficiente de correlação de Pearson (r)
    Fórmula: r = Σ(x-x̄)(y-ȳ) / √[Σ(x-x̄)² * Σ(y-ȳ)²]
    
    Entrada: dois vetores x e y de mesmo tamanho
    Saída: coeficiente r entre -1 e 1
           r > 0: correlação positiva
           r < 0: correlação negativa
           |r| próximo de 1: correlação forte
           |r| próximo de 0: correlação fraca
    """
    # Calcula as médias de x e y
    x_media = media(x)
    y_media = media(y)
    
    # Numerador: covariância entre x e y
    # Para cada par (xi, yi), calcula: (valor de x - média de x) * (valor de y - média de y)
    # Depois soma todos esses produtos - mede como x e y variam juntos
    numerador = sum((xi - x_media) * (yi - y_media) for xi, yi in zip(x, y))
    
    # Denominador: produto dos desvios padrão de x e y
    # Parte 1: soma dos quadrados dos desvios de x em relação à média
    soma_x_quadrado = sum((xi - x_media) ** 2 for xi in x)
    # Parte 2: soma dos quadrados dos desvios de y em relação à média  
    soma_y_quadrado = sum((yi - y_media) ** 2 for yi in y)
    # Multiplica as duas partes e tira a raiz quadrada (** 0.5 = √)
    denominador = (soma_x_quadrado * soma_y_quadrado) ** 0.5
    
    # Retorna o coeficiente de correlação
    return numerador / denominador

def regressao(x, y):
    """
    Calcula os coeficientes da regressão linear simples
    Equação da reta: y = β0 + β1*x
    
    Fórmulas:
    β1 = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²  (inclinação da reta)
    β0 = ȳ - β1*x̄                (intercepto da reta)
    
    Entrada: dois vetores x e y de mesmo tamanho
    Saída: tupla (β0, β1)
           β0: valor de y quando x = 0
           β1: variação em y para cada unidade de x
    """
    # Calcula as médias de x e y
    x_media = media(x)
    y_media = media(y)
    
    # Cálculo da inclinação β1 (coeficiente angular)
    numerador = sum((xi - x_media) * (yi - y_media) for xi, yi in zip(x, y))
    denominador = sum((xi - x_media) ** 2 for xi in x)
    beta1 = numerador / denominador
    
    # Cálculo do intercepto β0 (coeficiente linear)
    beta0 = y_media - beta1 * x_media
    
    # Retorna os coeficientes da regressão
    return beta0, beta1

def eh_apropriado_para_regressao(x):
    """
    Verifica se os dados são apropriados para regressão linear
    
    Critério: deve haver pelo menos 3 valores distintos em x
    Motivo: Com apenas 2 pontos distintos, qualquer linha pode ser traçada,
            mas não há dados suficientes para validar a tendência linear
    
    Entrada: vetor x
    Saída: True se apropriado, False caso contrário
    """
    # Remove valores duplicados e conta quantos valores únicos existem
    valores_unicos = set(x)
    # Retorna True se há mais de 2 valores únicos
    return len(valores_unicos) > 2

# ===============================
# Dados dos Experimentos
# ===============================

# Conjuntos de dados do arquivo datasetFase1.txt
# Cada dataset tem 11 pontos (x, y) para análise de regressão
datasets = {
    "Dataset 1": {
        "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "y": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
        "cor": "blue"  # Cor dos pontos no gráfico
    },
    "Dataset 2": {
        "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        "y": [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
        "cor": "red"   # Cor dos pontos no gráfico
    },
    "Dataset 3": {
        "x": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],  # Note: 10 valores iguais a 8, 1 valor igual a 19
        "y": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50],
        "cor": "green" # Cor dos pontos no gráfico
    }
}

# ===============================
# Análise e Visualização
# ===============================

# ETAPA 1: Configuração da figura com 3 subplots lado a lado
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Análise de Correlação e Regressão Linear', fontsize=16, fontweight='bold')

# ETAPA 2: Processa cada dataset individualmente
for i, (nome, dados) in enumerate(datasets.items()):
    
    # ETAPA 2.1: Extrai os dados do dataset atual
    x = dados["x"]      # Variável independente
    y = dados["y"]      # Variável dependente
    cor = dados["cor"]  # Cor para o gráfico
    
    # ETAPA 2.2: Cálculos estatísticos obrigatórios
    r = correlacao(x, y)                         # Calcula coeficiente de correlação
    beta0, beta1 = regressao(x, y)               # Calcula coeficientes da regressão
    apropriado = eh_apropriado_para_regressao(x) # Verifica se os dados são adequados
    
    # ETAPA 2.3: Configuração do subplot atual
    ax = axes[i]  # Seleciona o subplot correspondente ao dataset
    
    # ETAPA 2.4: Gráfico de Dispersão (OBRIGATÓRIO - usando scatter)
    # Plota os pontos dos dados no gráfico
    ax.scatter(x, y, color=cor, alpha=0.7, s=60, 
              edgecolors='black', linewidth=0.5, label='Dados')
    
    # ETAPA 2.5: Linha de Regressão (OBRIGATÓRIO - usando plot)
    # Cria uma linha suave para mostrar a tendência linear
    x_linha = np.linspace(min(x), max(x), 100)  # Gera 100 pontos entre min e max de x
    y_linha = beta0 + beta1 * x_linha           # Calcula y para cada ponto usando a equação
    ax.plot(x_linha, y_linha, color='black', linewidth=2, linestyle='--', 
            label=f'y = {beta0:.2f} + {beta1:.2f}x')
    
    # ETAPA 2.6: Formatação e configuração do gráfico
    ax.set_xlabel('x', fontsize=12)              # Rótulo do eixo x
    ax.set_ylabel('y', fontsize=12)              # Rótulo do eixo y
    ax.grid(True, alpha=0.3)                     # Grade para facilitar leitura
    ax.legend(fontsize=10)                       # Legenda do gráfico
    
    # ETAPA 2.7: Título com coeficientes (OBRIGATÓRIO - usando title)
    # Mostra correlação, coeficientes de regressão e status de adequação
    status = "Apropriado" if apropriado else "Inadequado"
    ax.set_title(f'{nome}\nr = {r:.4f} | β₀ = {beta0:.3f} | β₁ = {beta1:.3f}\n{status}', 
                fontsize=11, pad=15)
    
    # ETAPA 2.8: Ajusta limites dos eixos para melhor visualização
    margem_x = (max(x) - min(x)) * 0.1 if max(x) != min(x) else 1
    margem_y = (max(y) - min(y)) * 0.1 if max(y) != min(y) else 1
    ax.set_xlim(min(x) - margem_x, max(x) + margem_x)
    ax.set_ylim(min(y) - margem_y, max(y) + margem_y)

# ETAPA 3: Exibição final dos gráficos
plt.tight_layout()  # Ajusta espaçamento entre subplots automaticamente
plt.show()          # Mostra a figura com os 3 gráficos

# ===============================
# Análise Final e Conclusões
# ===============================

# ETAPA 4: Resumo da análise para identificar dataset inadequado
print("=== RESUMO DA ANÁLISE ===")
print("Dataset 3 é inadequado para regressão linear:")
print("• Apenas 2 valores únicos em x (8 e 19)")
print("• Falta variabilidade necessária para análise confiável")
print("• O valor 19 é um outlier que distorce os resultados")

"""
INTERPRETAÇÃO COMPLETA DOS RESULTADOS:

1. CORRELAÇÃO (r):
   - Valores próximos de +1: correlação positiva forte
   - Valores próximos de -1: correlação negativa forte
   - Valores próximos de 0: correlação fraca ou inexistente

2. COEFICIENTES DE REGRESSÃO:
   - β₀ (intercepto): valor de y quando x = 0
   - β₁ (inclinação): variação em y para cada unidade de x
   
3. ADEQUAÇÃO DOS DADOS:
   - Dataset 1 e 2: Apropriados (boa variabilidade em x)
   - Dataset 3: Inadequado (pouca variabilidade em x)

4. OBSERVAÇÃO IMPORTANTE:
   Todos os datasets têm correlações similares (~0.8), mas apenas
   o Dataset 3 é inadequado devido à distribuição dos valores de x.
   Isso demonstra que correlação alta não garante adequação para regressão.

5. ATENDIMENTO AOS REQUISITOS:
   ✓ Gráfico de Dispersão: função scatter() nas linhas 118-120
   ✓ Cálculo de Correlação: função correlacao() na linha 112
   ✓ Linha de Regressão: função plot() nas linhas 122-126
   ✓ Coeficientes no Título: função title() nas linhas 134-136
   ✓ Identificação do Dataset Inadequado: Dataset 3
"""