import matplotlib.pyplot as plt  # Para criação de gráficos
import numpy as np              # Para operações matemáticas e arrays
import pandas as pd             # Para manipulação de dados
from sklearn.metrics import r2_score, mean_squared_error  # Métricas de avaliação
from sklearn.model_selection import train_test_split     # Divisão treino/teste

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calcular_polinomio(x, coeficientes):
    """
    Calcula y = β₀ + β₁X + β₂X² + β₃X³ + ... + βₙXⁿ
    Nota: coeficientes do polyfit vêm em ordem reversa [βₙ, βₙ₋₁, ..., β₁, β₀]
    """
    grau = len(coeficientes) - 1                    # Determina o grau do polinômio
    y = np.zeros_like(x)                           # Inicializa array y com zeros
    
    for i, coef in enumerate(coeficientes):        # Para cada coeficiente
        potencia = grau - i                        # Calcula a potência correspondente
        y += coef * (x ** potencia)                # Adiciona o termo ao polinômio
    
    return y                                       # Retorna valores calculados

def exibir_equacao(coeficientes, grau):
    """Exibe a equação do polinômio de forma legível"""
    eq = "y = "                                    # Inicia string da equação
    for i, coef in enumerate(coeficientes):        # Para cada coeficiente
        potencia = grau - i                        # Calcula potência
        
        if i > 0:                                  # Se não é o primeiro termo
            eq += " + " if coef >= 0 else " - "    # Adiciona sinal
            coef = abs(coef)                       # Usa valor absoluto
        
        if potencia == 0:                          # Termo constante
            eq += f"{coef:.4f}"
        elif potencia == 1:                        # Termo linear
            eq += f"{coef:.4f}x"
        else:                                      # Termos de potência maior
            eq += f"{coef:.4f}x^{potencia}"
    
    return eq                                      # Retorna equação formatada

# =============================================================================
# a) CARREGAMENTO DOS DADOS
# =============================================================================

print("a) CARREGAMENTO DOS DADOS")              # Cabeçalho da seção
print("=" * 60)

# Tentativa de carregar arquivo .mat ou .csv
try:
    import scipy.io as scipy                       # Importa scipy para arquivos .mat
    mat = scipy.loadmat('data_preg.mat')          # Carrega arquivo .mat
    data = mat['data']                            # Extrai dados
    x_data = data[:, 0]                           # Primeira coluna = x
    y_data = data[:, 1]                           # Segunda coluna = y
    print("✓ Arquivo data_preg.mat carregado com sucesso")
except:                                           # Se falhar
    try:
        data = pd.read_csv('data_preg.csv', header=None)  # Tenta carregar .csv
        x_data = data.iloc[:, 0].values           # Primeira coluna = x
        y_data = data.iloc[:, 1].values           # Segunda coluna = y
        print("✓ Arquivo data_preg.csv carregado com sucesso")
    except:                                       # Se ambos falharem
        # Gerar dados sintéticos para demonstração
        print("⚠ Arquivos não encontrados. Gerando dados sintéticos...")
        np.random.seed(42)                        # Seed para reprodutibilidade
        x_data = np.linspace(0, 4, 20)           # 20 pontos entre 0 e 4
        y_data = 0.5 * x_data**3 - 2 * x_data**2 + x_data + 1 + np.random.normal(0, 0.5, 20)  # Função cúbica + ruído

print(f"✓ {len(x_data)} pontos carregados")      # Mostra quantidade de pontos
print(f"  X: min={x_data.min():.2f}, max={x_data.max():.2f}")  # Range de X
print(f"  Y: min={y_data.min():.2f}, max={y_data.max():.2f}")  # Range de Y

# =============================================================================
# b) GRÁFICO DE DISPERSÃO
# =============================================================================

print("\nb) GRÁFICO DE DISPERSÃO DOS DADOS")    # Cabeçalho da seção
print("=" * 60)

plt.figure(figsize=(12, 8))                      # Cria figura com tamanho específico
plt.scatter(x_data, y_data, alpha=0.7, s=60, color='blue', label='Dados Originais')  # Scatter plot
plt.xlabel('X')                                  # Rótulo eixo X
plt.ylabel('Y')                                  # Rótulo eixo Y
plt.title('Dados de Entrada - Gráfico de Dispersão')  # Título do gráfico
plt.legend()                                     # Mostra legenda
plt.grid(True, alpha=0.3)                       # Adiciona grade
plt.show()                                       # Exibe gráfico

# =============================================================================
# c-f) REGRESSÃO POLINOMIAL PARA DIFERENTES GRAUS
# =============================================================================

print("\nc-f) REGRESSÃO POLINOMIAL - DIFERENTES GRAUS")  # Cabeçalho da seção
print("=" * 60)

# Definir graus e cores
graus = [1, 2, 3, 8]                            # Graus a serem testados (N=1,2,3,8)
cores = ['red', 'green', 'black', 'yellow']     # Cores correspondentes
coeficientes_dict = {}                           # Dicionário para armazenar coeficientes
x_linha = np.linspace(x_data.min(), x_data.max(), 100)  # Pontos para linha suave

plt.figure(figsize=(14, 10))                    # Nova figura
plt.scatter(x_data, y_data, alpha=0.7, s=60, color='blue', label='Dados Originais')  # Scatter dos dados

for grau, cor in zip(graus, cores):             # Para cada grau e cor
    # Usar polyfit para obter coeficientes
    coeficientes = np.polyfit(x_data, y_data, grau)  # ✓ Calcula βs usando polyfit
    coeficientes_dict[grau] = coeficientes       # Armazena coeficientes
    
    # Calcular y usando nossa função personalizada
    y_linha = calcular_polinomio(x_linha, coeficientes)  # ✓ Não usa função pronta do Python
    
    # Plotar linha de regressão
    plt.plot(x_linha, y_linha, color=cor, linewidth=2,   # Plota linha de regressão
             label=f'Grau {grau} ({cor})')
    
    # Exibir equação
    print(f"\nGrau {grau}: {exibir_equacao(coeficientes, grau)}")  # Mostra equação

plt.xlabel('X')                                  # Rótulo eixo X
plt.ylabel('Y')                                  # Rótulo eixo Y
plt.title('Regressão Polinomial - Diferentes Graus')  # Título
plt.legend()                                     # Legenda
plt.grid(True, alpha=0.3)                       # Grade
plt.show()                                       # Exibe gráfico

# =============================================================================
# g) ERRO QUADRÁTICO MÉDIO (EQM) PARA CADA GRAU
# =============================================================================

print("\ng) ERRO QUADRÁTICO MÉDIO (EQM)")       # Cabeçalho da seção
print("=" * 60)

eqm_dict = {}                                    # Dicionário para armazenar EQM
for grau in graus:                               # Para cada grau
    coeficientes = coeficientes_dict[grau]       # Pega coeficientes
    y_pred = calcular_polinomio(x_data, coeficientes)  # Calcula predições
    eqm = mean_squared_error(y_data, y_pred)     # ✓ Calcula EQM
    eqm_dict[grau] = eqm                         # Armazena EQM
    print(f"Grau {grau}: EQM = {eqm:.6f}")       # Mostra EQM

melhor_grau = min(eqm_dict, key=eqm_dict.get)   # Encontra grau com menor EQM
print(f"\n✓ Modelo mais preciso (menor EQM): Grau {melhor_grau}")

# =============================================================================
# h) DIVISÃO DOS DADOS - TREINO E TESTE
# =============================================================================

print("\nh) DIVISÃO DOS DADOS (90% TREINO, 10% TESTE)")  # Cabeçalho da seção
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(     # ✓ Divisão 90% treino, 10% teste
    x_data, y_data, test_size=0.1, random_state=42
)

print(f"✓ Dados de treino: {len(X_train)} pontos")      # Mostra quantidade treino
print(f"✓ Dados de teste: {len(X_test)} pontos")        # Mostra quantidade teste

# =============================================================================
# i) REGRESSÃO USANDO APENAS DADOS DE TREINO
# =============================================================================

print("\ni) REGRESSÃO COM DADOS DE TREINO")     # Cabeçalho da seção
print("=" * 60)

coeficientes_treino = {}                         # Dicionário para coeficientes do treino
x_linha_treino = np.linspace(x_data.min(), x_data.max(), 100)  # Pontos para linha

plt.figure(figsize=(14, 10))                    # Nova figura
plt.scatter(X_train, y_train, alpha=0.7, s=60, color='blue', label='Dados de Treino')    # Scatter treino
plt.scatter(X_test, y_test, alpha=0.7, s=60, color='orange', label='Dados de Teste')     # Scatter teste

for grau, cor in zip(graus, cores):             # Para cada grau
    # Ajustar modelo apenas com dados de treino
    coeficientes = np.polyfit(X_train, y_train, grau)  # ✓ Usa apenas dados de treino
    coeficientes_treino[grau] = coeficientes     # Armazena coeficientes
    
    # Calcular linha de regressão
    y_linha = calcular_polinomio(x_linha_treino, coeficientes)  # Calcula linha
    
    # Plotar
    plt.plot(x_linha_treino, y_linha, color=cor, linewidth=2,   # Plota linha
             label=f'Grau {grau} (Treino)')

plt.xlabel('X')                                  # Rótulo eixo X
plt.ylabel('Y')                                  # Rótulo eixo Y
plt.title('Regressão Polinomial - Treinada apenas com Dados de Treino')  # Título
plt.legend()                                     # Legenda
plt.grid(True, alpha=0.3)                       # Grade
plt.show()                                       # Exibe gráfico

# =============================================================================
# j) EQM USANDO APENAS DADOS DE TESTE
# =============================================================================

print("\nj) EQM CALCULADO COM DADOS DE TESTE")  # Cabeçalho da seção
print("=" * 60)

eqm_teste_dict = {}                              # Dicionário para EQM do teste
for grau in graus:                               # Para cada grau
    coeficientes = coeficientes_treino[grau]     # Pega coeficientes do treino
    y_pred_teste = calcular_polinomio(X_test, coeficientes)  # ✓ Predições apenas no teste
    eqm_teste = mean_squared_error(y_test, y_pred_teste)     # ✓ EQM apenas no teste
    eqm_teste_dict[grau] = eqm_teste             # Armazena EQM
    print(f"Grau {grau}: EQM (teste) = {eqm_teste:.6f}")    # Mostra EQM

melhor_grau_teste = min(eqm_teste_dict, key=eqm_teste_dict.get)  # Melhor grau no teste
print(f"\n✓ Modelo mais preciso no teste: Grau {melhor_grau_teste}")

# =============================================================================
# k) COEFICIENTE DE DETERMINAÇÃO (R²)
# =============================================================================

print("\nk) COEFICIENTE DE DETERMINAÇÃO (R²)")  # Cabeçalho da seção
print("=" * 60)

print(f"{'Grau':<6} {'R² Treino':<12} {'R² Teste':<12} {'EQM Treino':<12} {'EQM Teste'}")  # Cabeçalho tabela
print("-" * 60)

for grau in graus:                               # Para cada grau
    coeficientes = coeficientes_treino[grau]     # Pega coeficientes
    
    # Predições
    y_pred_treino = calcular_polinomio(X_train, coeficientes)  # Predições treino
    y_pred_teste = calcular_polinomio(X_test, coeficientes)    # Predições teste
    
    # R² scores
    r2_treino = r2_score(y_train, y_pred_treino)    # ✓ R² do treino
    r2_teste = r2_score(y_test, y_pred_teste)       # ✓ R² do teste
    
    # EQM
    eqm_treino = mean_squared_error(y_train, y_pred_treino)    # EQM treino
    eqm_teste = mean_squared_error(y_test, y_pred_teste)       # EQM teste
    
    print(f"{grau:<6} {r2_treino:<12.4f} {r2_teste:<12.4f} {eqm_treino:<12.6f} {eqm_teste:<12.6f}")  # Mostra resultados

# =============================================================================
# l) ANÁLISE E CONCLUSÕES
# =============================================================================

print("\nl) ANÁLISE E CONCLUSÕES")              # Cabeçalho da seção
print("=" * 60)

print("OBSERVAÇÕES SOBRE OVERFITTING:")         # Análise de overfitting
print("━" * 40)

# Análise de overfitting
print("\n1. COMPARAÇÃO R² TREINO vs TESTE:")
for grau in graus:                               # Para cada grau
    coeficientes = coeficientes_treino[grau]     # Pega coeficientes
    y_pred_treino = calcular_polinomio(X_train, coeficientes)  # Predições treino
    y_pred_teste = calcular_polinomio(X_test, coeficientes)    # Predições teste
    
    r2_treino = r2_score(y_train, y_pred_treino)    # R² treino
    r2_teste = r2_score(y_test, y_pred_teste)       # R² teste
    
    diferenca = r2_treino - r2_teste             # Diferença entre R²
    
    if diferenca > 0.1:                          # Critério para overfitting
        status = "🔴 POSSÍVEL OVERFITTING"
    elif diferenca > 0.05:
        status = "🟡 ATENÇÃO"
    else:
        status = "🟢 OK"
    
    print(f"   Grau {grau}: Δ = {diferenca:.4f} {status}")  # Mostra análise

# Restante do código continua com mais análises e conclusões...