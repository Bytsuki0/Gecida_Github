import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analisar_dataframe(caminho_csv):
    try:
        # 1. Carregar o arquivo CSV
        df = pd.read_csv(caminho_csv)
        print("📊 Arquivo carregado com sucesso!\n")
        
        # --- DESCRIÇÃO DOS DADOS ---
        print("=== 1. VISÃO GERAL DOS DADOS ===")
        print(f"Total de Linhas: {df.shape[0]}")
        print(f"Total de Colunas: {df.shape[1]}\n")
        
        print("--- Primeiras 5 linhas do dataset ---")
        print(df.head(), "\n")
        
        print("--- Informações Tipos de Dados ---")
        print(df.info(), "\n")
        
        print("--- Estatísticas Descritivas (Variáveis Numéricas) ---")
        print(df.describe(), "\n")
        
        # --- DADOS AUSENTES ---
        print("=== 2. DADOS AUSENTES / FALTANTES ===")
        valores_ausentes = df.isnull().sum()
        porcentagem_ausentes = (df.isnull().sum() / len(df)) * 100
        
        # Criando um DataFrame para exibir os dados faltantes de forma limpa
        df_faltantes = pd.DataFrame({
            'Total Ausentes': valores_ausentes,
            'Porcentagem (%)': porcentagem_ausentes
        })
        # Filtra apenas as colunas que possuem pelo menos 1 dado ausente
        df_faltantes = df_faltantes[df_faltantes['Total Ausentes'] > 0].sort_values(by='Total Ausentes', ascending=False)
        
        if df_faltantes.empty:
            print("🎉 Excelente! Nenhum dado ausente encontrado.\n")
        else:
            print(df_faltantes, "\n")
            
        # --- CORRELAÇÃO ---
        print("=== 3. MATRIZ DE CORRELAÇÃO ===")
        # Seleciona apenas colunas numéricas para calcular a correlação
        df_numerico = df.select_dtypes(include=['number'])
        
        if df_numerico.shape[1] > 1:
            matriz_correlacao = df_numerico.corr()
            print(matriz_correlacao, "\n")
            
            # Gerando um mapa de calor (Heatmap) da correlação
            print("Generating heatmap de correlação...")
            plt.figure(figsize=(10, 8))
            sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title('Matriz de Correlação')
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Não há colunas numéricas suficientes para calcular correlação.\n")
            
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo no caminho '{caminho_csv}' não foi encontrado. Verifique o nome e tente novamente.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado: {e}")

# --- COMO USAR ---
# Substitua pelo caminho do seu arquivo CSV (ex: 'dados.csv' ou 'C:/pasta/dados.csv')
caminho_do_seu_arquivo = 'seu_arquivo.csv' 

# Executa a função
analisar_dataframe(caminho_do_seu_arquivo)