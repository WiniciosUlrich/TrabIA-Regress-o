import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calcular_polinomio(x, coeficientes):
    """
    Calcula y = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ
    Nota: coeficientes do polyfit vêm em ordem reversa [βₙ, βₙ₋₁, ..., β₁, β₀]
    """
    grau = len(coeficientes) - 1
    y = np.zeros_like(x)
    
    for i, coef in enumerate(coeficientes):
        potencia = grau - i
        y += coef * (x ** potencia)
    
    return y

def exibir_equacao(coeficientes, grau):
    """Exibe a equação do polinômio de forma legível"""
    eq = "y = "
    for i, coef in enumerate(coeficientes):
        potencia = grau - i
        
        if i > 0:
            eq += " + " if coef >= 0 else " - "
            coef = abs(coef)
        
        if potencia == 0:
            eq += f"{coef:.4f}"
        elif potencia == 1:
            eq += f"{coef:.4f}x"
        else:
            eq += f"{coef:.4f}x^{potencia}"
    
    return eq

# =============================================================================
# a) CARREGAMENTO DOS DADOS
# =============================================================================

print("a) CARREGAMENTO DOS DADOS")
print("=" * 60)

# Tentativa de carregar arquivo .mat ou .csv
try:
    import scipy.io as scipy
    mat = scipy.loadmat('data_preg.mat')
    data = mat['data']
    x_data = data[:, 0]
    y_data = data[:, 1]
    print("✓ Arquivo data_preg.mat carregado com sucesso")
except:
    try:
        data = pd.read_csv('data_preg.csv', header=None)
        x_data = data.iloc[:, 0].values
        y_data = data.iloc[:, 1].values
        print("✓ Arquivo data_preg.csv carregado com sucesso")
    except:
        # Gerar dados sintéticos para demonstração
        print("⚠ Arquivos não encontrados. Gerando dados sintéticos...")
        np.random.seed(42)
        x_data = np.linspace(0, 4, 20)
        y_data = 0.5 * x_data**3 - 2 * x_data**2 + x_data + 1 + np.random.normal(0, 0.5, 20)

print(f"✓ {len(x_data)} pontos carregados")
print(f"  X: min={x_data.min():.2f}, max={x_data.max():.2f}")
print(f"  Y: min={y_data.min():.2f}, max={y_data.max():.2f}")

# =============================================================================
# b) GRÁFICO DE DISPERSÃO
# =============================================================================

print("\nb) GRÁFICO DE DISPERSÃO DOS DADOS")
print("=" * 60)

plt.figure(figsize=(12, 8))
plt.scatter(x_data, y_data, alpha=0.7, s=60, color='blue', label='Dados Originais')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dados de Entrada - Gráfico de Dispersão')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# c-f) REGRESSÃO POLINOMIAL PARA DIFERENTES GRAUS
# =============================================================================

print("\nc-f) REGRESSÃO POLINOMIAL - DIFERENTES GRAUS")
print("=" * 60)

# Definir graus e cores
graus = [1, 2, 3, 8]
cores = ['red', 'green', 'black', 'yellow']
coeficientes_dict = {}
x_linha = np.linspace(x_data.min(), x_data.max(), 100)

plt.figure(figsize=(14, 10))
plt.scatter(x_data, y_data, alpha=0.7, s=60, color='blue', label='Dados Originais')

for grau, cor in zip(graus, cores):
    # Usar polyfit para obter coeficientes
    coeficientes = np.polyfit(x_data, y_data, grau)
    coeficientes_dict[grau] = coeficientes
    
    # Calcular y usando nossa função personalizada
    y_linha = calcular_polinomio(x_linha, coeficientes)
    
    # Plotar linha de regressão
    plt.plot(x_linha, y_linha, color=cor, linewidth=2, 
             label=f'Grau {grau} ({cor})')
    
    # Exibir equação
    print(f"\nGrau {grau}: {exibir_equacao(coeficientes, grau)}")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regressão Polinomial - Diferentes Graus')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# g) ERRO QUADRÁTICO MÉDIO (EQM) PARA CADA GRAU
# =============================================================================

print("\ng) ERRO QUADRÁTICO MÉDIO (EQM)")
print("=" * 60)

eqm_dict = {}
for grau in graus:
    coeficientes = coeficientes_dict[grau]
    y_pred = calcular_polinomio(x_data, coeficientes)
    eqm = mean_squared_error(y_data, y_pred)
    eqm_dict[grau] = eqm
    print(f"Grau {grau}: EQM = {eqm:.6f}")

melhor_grau = min(eqm_dict, key=eqm_dict.get)
print(f"\n✓ Modelo mais preciso (menor EQM): Grau {melhor_grau}")

# =============================================================================
# h) DIVISÃO DOS DADOS - TREINO E TESTE
# =============================================================================

print("\nh) DIVISÃO DOS DADOS (90% TREINO, 10% TESTE)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.1, random_state=42
)

print(f"✓ Dados de treino: {len(X_train)} pontos")
print(f"✓ Dados de teste: {len(X_test)} pontos")

# =============================================================================
# i) REGRESSÃO USANDO APENAS DADOS DE TREINO
# =============================================================================

print("\ni) REGRESSÃO COM DADOS DE TREINO")
print("=" * 60)

coeficientes_treino = {}
x_linha_treino = np.linspace(x_data.min(), x_data.max(), 100)

plt.figure(figsize=(14, 10))
plt.scatter(X_train, y_train, alpha=0.7, s=60, color='blue', label='Dados de Treino')
plt.scatter(X_test, y_test, alpha=0.7, s=60, color='orange', label='Dados de Teste')

for grau, cor in zip(graus, cores):
    # Ajustar modelo apenas com dados de treino
    coeficientes = np.polyfit(X_train, y_train, grau)
    coeficientes_treino[grau] = coeficientes
    
    # Calcular linha de regressão
    y_linha = calcular_polinomio(x_linha_treino, coeficientes)
    
    # Plotar
    plt.plot(x_linha_treino, y_linha, color=cor, linewidth=2, 
             label=f'Grau {grau} (Treino)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regressão Polinomial - Treinada apenas com Dados de Treino')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================================================================
# j) EQM USANDO APENAS DADOS DE TESTE
# =============================================================================

print("\nj) EQM CALCULADO COM DADOS DE TESTE")
print("=" * 60)

eqm_teste_dict = {}
for grau in graus:
    coeficientes = coeficientes_treino[grau]
    y_pred_teste = calcular_polinomio(X_test, coeficientes)
    eqm_teste = mean_squared_error(y_test, y_pred_teste)
    eqm_teste_dict[grau] = eqm_teste
    print(f"Grau {grau}: EQM (teste) = {eqm_teste:.6f}")

melhor_grau_teste = min(eqm_teste_dict, key=eqm_teste_dict.get)
print(f"\n✓ Modelo mais preciso no teste: Grau {melhor_grau_teste}")

# =============================================================================
# k) COEFICIENTE DE DETERMINAÇÃO (R²)
# =============================================================================

print("\nk) COEFICIENTE DE DETERMINAÇÃO (R²)")
print("=" * 60)

print(f"{'Grau':<6} {'R² Treino':<12} {'R² Teste':<12} {'EQM Treino':<12} {'EQM Teste'}")
print("-" * 60)

for grau in graus:
    coeficientes = coeficientes_treino[grau]
    
    # Predições
    y_pred_treino = calcular_polinomio(X_train, coeficientes)
    y_pred_teste = calcular_polinomio(X_test, coeficientes)
    
    # R² scores
    r2_treino = r2_score(y_train, y_pred_treino)
    r2_teste = r2_score(y_test, y_pred_teste)
    
    # EQM
    eqm_treino = mean_squared_error(y_train, y_pred_treino)
    eqm_teste = mean_squared_error(y_test, y_pred_teste)
    
    print(f"{grau:<6} {r2_treino:<12.4f} {r2_teste:<12.4f} {eqm_treino:<12.6f} {eqm_teste:<12.6f}")

# =============================================================================
# l) ANÁLISE E CONCLUSÕES
# =============================================================================

print("\nl) ANÁLISE E CONCLUSÕES")
print("=" * 60)

print("OBSERVAÇÕES SOBRE OVERFITTING:")
print("━" * 40)

# Análise de overfitting
print("\n1. COMPARAÇÃO R² TREINO vs TESTE:")
for grau in graus:
    coeficientes = coeficientes_treino[grau]
    y_pred_treino = calcular_polinomio(X_train, coeficientes)
    y_pred_teste = calcular_polinomio(X_test, coeficientes)
    
    r2_treino = r2_score(y_train, y_pred_treino)
    r2_teste = r2_score(y_test, y_pred_teste)
    
    diferenca = r2_treino - r2_teste
    
    if diferenca > 0.1:
        status = "🔴 POSSÍVEL OVERFITTING"
    elif diferenca > 0.05:
        status = "🟡 ATENÇÃO"
    else:
        status = "🟢 OK"
    
    print(f"   Grau {grau}: Δ = {diferenca:.4f} {status}")

print("\n2. MODELO MAIS PRECISO:")
print(f"   • Melhor no treino (R²): Grau {max(graus, key=lambda g: r2_score(y_train, calcular_polinomio(X_train, coeficientes_treino[g])))}")
print(f"   • Melhor no teste (R²): Grau {max(graus, key=lambda g: r2_score(y_test, calcular_polinomio(X_test, coeficientes_treino[g])))}")
print(f"   • Menor EQM no teste: Grau {melhor_grau_teste}")

print("\n3. CONCLUSÃO:")
print("   O modelo de grau mais alto pode ter melhor desempenho nos dados")
print("   de treino, mas pode ter pior generalização (overfitting).")
print("   O modelo ideal equilibra precisão e capacidade de generalização.")

# Gráfico comparativo final
plt.figure(figsize=(12, 8))

graus_plot = list(range(1, 9))
r2_treino_plot = []
r2_teste_plot = []

for g in graus_plot:
    if g in graus:
        coef = coeficientes_treino[g]
    else:
        coef = np.polyfit(X_train, y_train, g)
    
    y_pred_tr = calcular_polinomio(X_train, coef)
    y_pred_te = calcular_polinomio(X_test, coef)
    
    r2_treino_plot.append(r2_score(y_train, y_pred_tr))
    r2_teste_plot.append(r2_score(y_test, y_pred_te))

plt.plot(graus_plot, r2_treino_plot, 'o-', color='blue', label='R² Treino', linewidth=2)
plt.plot(graus_plot, r2_teste_plot, 's-', color='red', label='R² Teste', linewidth=2)
plt.xlabel('Grau do Polinômio')
plt.ylabel('Coeficiente de Determinação (R²)')
plt.title('Comparação R² - Treino vs Teste (Detecção de Overfitting)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(graus_plot)
plt.show()

print(f"\n✓ Script concluído! Modelo recomendado: Grau {melhor_grau_teste}")