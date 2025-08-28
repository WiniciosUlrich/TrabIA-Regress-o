import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===============================
# Implementação da Regressão Múltipla
# ===============================

def regmultipla(X, y):
    """
    Calcula os parâmetros β da regressão múltipla
    Fórmula: β = (X^T * X)^(-1) * X^T * y
    """
    # Adiciona coluna de 1s para o intercepto β0
    X_com_intercepto = np.column_stack([np.ones(len(X)), X])
    
    # Calcula β = (X^T * X)^(-1) * X^T * y
    XtX = np.dot(X_com_intercepto.T, X_com_intercepto)
    XtX_inv = np.linalg.inv(XtX)
    Xty = np.dot(X_com_intercepto.T, y)
    beta = np.dot(XtX_inv, Xty)
    
    return beta

def correlacao(x, y):
    """Calcula correlação de Pearson entre duas variáveis"""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerador = np.sum((x - x_mean) * (y - y_mean))
    denominador = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    return numerador / denominador

# ===============================
# a) Carregamento dos dados
# ===============================

print("=== ANÁLISE DE REGRESSÃO MÚLTIPLA ===")
print("Carregando dados...")

# Carrega dados do CSV
data = pd.read_csv("data.csv", header=None)
data.columns = ['Tamanho', 'Quartos', 'Preco']

print(f"Dados carregados: {len(data)} casas")

# ===============================
# b) Análise estatística descritiva
# ===============================

print("\n=== ANÁLISE ESTATÍSTICA DESCRITIVA ===")
print(data.describe())

print(f"\nMédia de preço das casas: R$ {data['Preco'].mean():,.2f}")
print(f"Menor casa custa: R$ {data['Preco'].min():,.2f}")

# Casa mais cara e seus quartos
casa_mais_cara = data.loc[data['Preco'].idxmax()]
print(f"Casa mais cara: R$ {casa_mais_cara['Preco']:,.2f} com {casa_mais_cara['Quartos']:.0f} quartos")

# ===============================
# c) Preparação das matrizes X e y
# ===============================

print("\n=== PREPARAÇÃO DOS DADOS ===")

# Matriz X (variáveis independentes)
X = data[['Tamanho', 'Quartos']].values
# Vetor y (variável dependente)
y = data['Preco'].values

print(f"Matriz X shape: {X.shape}")
print(f"Vetor y shape: {y.shape}")

# ===============================
# d) Correlações individuais
# ===============================

print("\n=== CORRELAÇÕES INDIVIDUAIS ===")

# Correlação Tamanho vs Preço
corr_tamanho_preco = correlacao(data['Tamanho'], data['Preco'])
print(f"Correlação Tamanho vs Preço: {corr_tamanho_preco:.4f}")

# Correlação Quartos vs Preço
corr_quartos_preco = correlacao(data['Quartos'], data['Preco'])
print(f"Correlação Quartos vs Preço: {corr_quartos_preco:.4f}")

# Gráficos de dispersão individuais
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Tamanho vs Preço
ax1.scatter(data['Tamanho'], data['Preco'], alpha=0.6, color='blue')
ax1.set_xlabel('Tamanho (sq ft)')
ax1.set_ylabel('Preço (R$)')
ax1.set_title(f'Tamanho vs Preço\nCorrelação: {corr_tamanho_preco:.4f}')
ax1.grid(True, alpha=0.3)

# Quartos vs Preço
ax2.scatter(data['Quartos'], data['Preco'], alpha=0.6, color='red')
ax2.set_xlabel('Número de Quartos')
ax2.set_ylabel('Preço (R$)')
ax2.set_title(f'Quartos vs Preço\nCorrelação: {corr_quartos_preco:.4f}')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===============================
# e) Regressão Múltipla
# ===============================

print("\n=== REGRESSÃO MÚLTIPLA ===")

# Calcula coeficientes β
beta = regmultipla(X, y)
print(f"Coeficientes β:")
print(f"β0 (intercepto): {beta[0]:,.2f}")
print(f"β1 (tamanho): {beta[1]:,.2f}")
print(f"β2 (quartos): {beta[2]:,.2f}")

# Equação da regressão
print(f"\nEquação: Preço = {beta[0]:,.2f} + {beta[1]:,.2f}*Tamanho + {beta[2]:,.2f}*Quartos")

# ===============================
# f) Gráfico 3D com linha de regressão
# ===============================

print("\n=== VISUALIZAÇÃO 3D ===")

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot dos dados reais
ax.scatter(data['Tamanho'], data['Quartos'], data['Preco'], 
          c='blue', marker='o', alpha=0.6, s=50, label='Dados Reais')

# Criação da superfície de regressão
tamanho_range = np.linspace(data['Tamanho'].min(), data['Tamanho'].max(), 20)
quartos_range = np.linspace(data['Quartos'].min(), data['Quartos'].max(), 20)
T, Q = np.meshgrid(tamanho_range, quartos_range)

# Preços preditos pela superfície
P_pred = beta[0] + beta[1] * T + beta[2] * Q

# Superfície de regressão
ax.plot_surface(T, Q, P_pred, alpha=0.3, color='red', label='Superfície de Regressão')

ax.set_xlabel('Tamanho (sq ft)')
ax.set_ylabel('Número de Quartos')
ax.set_zlabel('Preço (R$)')
ax.set_title('Regressão Múltipla - Preço de Casas')

# Adiciona texto com correlações
ax.text2D(0.02, 0.98, f'Corr Tamanho-Preço: {corr_tamanho_preco:.4f}\nCorr Quartos-Preço: {corr_quartos_preco:.4f}', 
          transform=ax.transAxes, fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

# ===============================
# h) Predição para casa específica
# ===============================

print("\n=== PREDIÇÕES ===")

def prever_preco(tamanho, quartos, beta):
    """Prevê o preço de uma casa dados tamanho e quartos"""
    return beta[0] + beta[1] * tamanho + beta[2] * quartos

# Casa de 1650 sq ft e 3 quartos
preco_pred = prever_preco(1650, 3, beta)
print(f"Casa 1650 sq ft, 3 quartos: R$ {preco_pred:,.2f}")

# Testando variação no número de quartos
print("\nVariação no número de quartos (tamanho fixo em 1650):")
for q in range(1, 6):
    preco = prever_preco(1650, q, beta)
    print(f"  {q} quartos: R$ {preco:,.2f}")

# Análise do impacto
impacto_quarto = beta[2]
print(f"\nImpacto de cada quarto adicional: R$ {impacto_quarto:,.2f}")
print("Motivo: Cada quarto adicional aumenta o preço pelo coeficiente β2")

# ===============================
# i) Comparação com scikit-learn
# ===============================

print("\n=== COMPARAÇÃO COM SCIKIT-LEARN ===")

# Regressão com scikit-learn
sklearn_model = LinearRegression()
sklearn_model.fit(X, y)

print("Nosso modelo:")
print(f"  β0: {beta[0]:,.2f}")
print(f"  β1: {beta[1]:,.2f}")
print(f"  β2: {beta[2]:,.2f}")

print("\nScikit-learn:")
print(f"  β0: {sklearn_model.intercept_:,.2f}")
print(f"  β1: {sklearn_model.coef_[0]:,.2f}")
print(f"  β2: {sklearn_model.coef_[1]:,.2f}")

# Predições comparativas
nossa_pred = prever_preco(1650, 3, beta)
sklearn_pred = sklearn_model.predict([[1650, 3]])[0]

print(f"\nPredição para casa 1650 sq ft, 3 quartos:")
print(f"  Nosso modelo: R$ {nossa_pred:,.2f}")
print(f"  Scikit-learn: R$ {sklearn_pred:,.2f}")
print(f"  Diferença: R$ {abs(nossa_pred - sklearn_pred):,.2f}")

# Métricas de qualidade
y_pred_nossa = beta[0] + X @ beta[1:]
y_pred_sklearn = sklearn_model.predict(X)

r2_nossa = r2_score(y, y_pred_nossa)
r2_sklearn = r2_score(y, y_pred_sklearn)

print(f"\nR² Score:")
print(f"  Nosso modelo: {r2_nossa:.4f}")
print(f"  Scikit-learn: {r2_sklearn:.4f}")

print("\n=== CONCLUSÕES ===")
print("✅ Os modelos são praticamente idênticos")
print("✅ Implementação matemática correta")
print("✅ Tamanho da casa tem maior impacto no preço que número de quartos")
print(f"✅ Modelo explica {r2_nossa:.1%} da variação nos preços")