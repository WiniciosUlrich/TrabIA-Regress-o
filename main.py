# ===============================
# Funções Estatísticas Básicas
# ===============================

def media(vetor):
    """Calcula a média aritmética de um conjunto de dados"""
    return sum(vetor) / len(vetor)

def correlacao(x, y):
    """
    Calcula o coeficiente de correlação de Pearson (r)
    Fórmula: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² * Σ(yi - ȳ)²]
    Retorna valor entre -1 (correlação negativa perfeita) e 1 (correlação positiva perfeita)
    """
    xm = media(x)  # Média de x
    ym = media(y)  # Média de y
    
    # Numerador: covariância entre x e y
    # Para cada par (xi, yi), calcula: (valor de x - média de x) * (valor de y - média de y)
    # Depois soma todos esses produtos - mede como x e y variam juntos
    numerador = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    
    # Denominador: produto dos desvios padrão de x e y
    # Parte 1: soma dos quadrados dos desvios de x em relação à média
    # Parte 2: soma dos quadrados dos desvios de y em relação à média  
    # Multiplica as duas partes e tira a raiz quadrada (** 0.5 = √)
    denominador = (sum((xi - xm) ** 2 for xi in x) * sum((yi - ym) ** 2 for yi in y)) ** 0.5
    
    return numerador / denominador

def regressao(x, y):
    """
    Calcula os coeficientes da regressão linear: y = β0 + β1*x
    Retorna: (β0, β1) onde β0 = intercepto e β1 = inclinação
    """
    xm = media(x)
    ym = media(y)
    
    # β1 = Σ[(xi - x̄)(yi - ȳ)] / Σ(xi - x̄)²
    numerador = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, y))
    denominador = sum((xi - xm) ** 2 for xi in x)
    b1 = numerador / denominador  # Inclinação da reta
    
    # β0 = ȳ - β1*x̄ (intercepto)
    b0 = ym - b1 * xm
    
    return b0, b1

def eh_apropriado_para_regressao(x):
    """
    Verifica se os dados são apropriados para regressão linear
    Critério: deve haver pelo menos 3 valores distintos em x
    """
    return len(set(x)) > 2

# ===============================
# Conjuntos de Dados para Análise
# ===============================

# Os datasets representam diferentes cenários de correlação
datasets = {
    "Dataset 1": (
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], 
        [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68] 
    ),
    "Dataset 2": (
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],  
        [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]  
    ),
    "Dataset 3": (
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],  
        [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]
    )
}

# ===============================
# Processamento e Análise
# ===============================

print("=== ANÁLISE DE REGRESSÃO LINEAR ===")

for nome, (x, y) in datasets.items():
    # Calcula estatísticas para cada dataset
    r = correlacao(x, y)              # Coeficiente de correlação
    b0, b1 = regressao(x, y)         # Coeficientes da reta de regressão
    apropriado = eh_apropriado_para_regressao(x)  # Validação dos dados
    
    print(f"\n>> {nome}")
    print(f"Correlação (r): {r:.4f}")
    print(f"Equação da reta: y = {b0:.4f} + {b1:.4f}*x")
    print(f"Coeficiente β0 (intercepto): {b0:.4f}")
    print(f"Coeficiente β1 (inclinação): {b1:.4f}")
    print("Apropriado para regressão linear?", "Sim" if apropriado else "Não")
# ===============================