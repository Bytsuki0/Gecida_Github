import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página para aproveitar melhor o espaço na tela
st.set_page_config(
    page_title="Pipeline de Predição - Listening Time",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 Protótipo do Pipeline de Desenvolvimento")
st.markdown("---")

# Criando as abas com base nas 6 fases do seu diagrama UML
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📥 Ingestão de Dados (Fase 1)",
    "📊 Análise Exploratória (Fase 2)",
    "🧼 Pré-processamento (Fase 3)",
    "🧠 Modelagem (Fase 4)",
    "🎯 Validação & Submissão (Fase 5)",
    "📄 Relatório LaTeX (Fase 6)"
])

# ==============================================================================
# FASE 1 - CARREGAMENTO DE DADOS
# ==============================================================================
with tab1:
    st.header("Fase 1 - Carregamento de Dados")
    st.write("Inspeção inicial dos conjuntos de treino e teste.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload do arquivo train.csv")
        file_train = st.file_uploader("Escolha o arquivo de treino", type=["csv"], key="train")
        # Simulação de feedback visual
        st.metric(label="Shape de Treino Simulado", value="15,000 linhas x 12 colunas")
        
    with col2:
        st.subheader("Upload do arquivo test.csv")
        file_test = st.file_uploader("Escolha o arquivo de teste", type=["csv"], key="test")
        st.metric(label="Shape de Teste Simulado", value="5,000 linhas x 11 colunas")

    st.info("💡 Código executado no backend: `pd.read_csv()` seguido de inspeção de integridade.")

# ==============================================================================
# FASE 2 - ANÁLISE EXPLORATÓRIA (EDA)
# ==============================================================================
with tab2:
    st.header("Fase 2 - Análise Exploratória (EDA)")
    
    eda_option = st.selectbox(
        "Selecione a visualização da EDA:",
        ["Descrição das Variáveis", "Distribuição da Variável Alvo", "Análise Numérica/Categórica", "Matriz de Correlação"]
    )
    
    if eda_option == "Descrição das Variáveis":
        st.subheader("Metadados e Cardinalidade")
        st.write("Exibição de `df.describe()`, dtypes e contagem de valores nulos (% NaN).")
        # Mock de tabela de propriedades
        mock_desc = pd.DataFrame({
            'Coluna': ['id', 'Genre', 'Length', 'Host_Pop', 'Listening_Time_minutes'],
            'Tipo': ['int64', 'object', 'float64', 'int64', 'float64'],
            '% NaN': ['0%', '2.1%', '0.5%', '0%', '0%']
        })
        st.dataframe(mock_desc, use_container_width=True)
        
    elif eda_option == "Distribuição da Variável Alvo":
        st.subheader("Distribuição de: Listening_Time_minutes")
        st.info("Aqui serão renderizados o Histograma e Boxplot para avaliar Skewness e Kurtosis.")
        # Espaço reservado para o gráfico
        st.bar_chart(np.random.exponential(scale=30.0, size=100))
        
    elif eda_option == "Análise Numérica/Categórica":
        st.subheader("Relações de Variáveis e Outliers (IQR)")
        st.write("Análise de dispersão (Scatter plots vs Alvo) e impacto de variáveis como `Is_Weekend` e interações.")
        
    elif eda_option == "Matriz de Correlação":
        st.subheader("Heatmap de Pearson e Spearman")
        st.caption("Renderização via Seaborn / Plotly mapeando correlações mais fortes com a variável alvo.")

# ==============================================================================
# FASE 3 - PRÉ-PROCESSAMENTO
# ==============================================================================
with tab3:
    st.header("Fase 3 - Pré-processamento")
    st.write("Configuração das regras de negócio para tratamento do dataset:")
    
    col_nan, col_outliers = st.columns(2)
    
    with col_nan:
        st.subheader("Tratamento de Valores Ausentes (NaN)")
        st.warning("Regra UML: Drop coluna se > 50% NaN | Drop linha se > 25% NaN")
        st.checkbox("Aplicar Imputador KNN para NaNs restantes (Mediana/Moda)", value=True)
        
    with col_outliers:
        st.subheader("Engenharia de Recursos & Escalonamento")
        st.selectbox("Método de Tratamento de Outliers", ["IQR Clipping", "Winsorization", "Z-Score (> 3)"])
        st.selectbox("Encoding Categórico", ["Ordinal (Pub_Time)", "One-Hot Encoding (Genre, Day)"])
        st.selectbox("Escalonamento", ["StandardScaler", "MinMaxScaler"])

    st.subheader("Seleção Final de Features")
    st.write("Filtros aplicados: Multicolinearidade (VIF), Correlação > 0.9 e Feature Importance.")

# ==============================================================================
# FASE 4 - MODELAGEM
# ==============================================================================
with tab4:
    st.header("Fase 4 - Modelagem")
    st.write("Validação Cruzada: K-Fold (k=5) Estratificado.")
    
    st.subheader("Modelos Disponíveis (Estrutura Fork do UML)")
    modelos_selecionados = st.multiselect(
        "Selecione quais algoritmos treinar em paralelo:",
        ["Linear Regression (Baseline)", "Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor"],
        default=["Linear Regression (Baseline)", "XGBoost Regressor"]
    )
    
    st.subheader("Comparativo de Métricas (Simulado)")
    # Tabela comparativa simulada
    metrics_df = pd.DataFrame({
        'Modelo': ['Linear Regression', 'Random Forest', 'XGBoost'],
        'RMSE': [12.45, 8.21, 6.12],
        'MAE': [9.30, 6.10, 4.25],
        'R²': [0.55, 0.78, 0.89],
        'RMSLE': [0.31, 0.22, 0.15]
    })
    st.table(metrics_df)
    
    st.subheader("Otimização & Ensembles")
    st.write("Tuning via **Optuna** / **GridSearch** associado ao modelo final por **VotingRegressor**.")

# ==============================================================================
# FASE 5 - VALIDAÇÃO COM TEST.CSV
# ==============================================================================
with tab5:
    st.header("Fase 5 - Validação com test.csv")
    st.write("Aplicação do melhor pipeline treinado sobre os dados invisíveis de teste.")
    
    if st.button("🚀 Executar Predição Final"):
        st.success("Predição realizada com sucesso usando o melhor pipeline!")
        
        st.subheader("Visualização do arquivo de submissão (submission.csv)")
        # Gerando um dataframe simulado para exibição
        mock_submission = pd.DataFrame({
            'id': range(1001, 1006),
            'Listening_Time_minutes': [24.5, 112.3, 45.1, 0.0, 78.9]
        })
        st.dataframe(mock_submission, use_container_width=True)
        
        st.download_button(
            label="💾 Baixar submission.csv",
            data=mock_submission.to_csv(index=False),
            file_name="submission.csv",
            mime="text/csv"
        )

# ==============================================================================
# FASE 6 - RELATÓRIO LATEX / PDF
# ==============================================================================
with tab6:
    st.header("Fase 6 - Relatório LaTeX / PDF")
    st.write("Etapa de compilação dos artefatos científicos para geração do documento final.")
    
    st.markdown("""
    * **Módulos Incluídos na Compilação:**
        * Tabelas dinâmicas de métricas geradas na Fase 4.
        * Gráficos exportados da Análise Exploratória (Fase 2).
        * Conclusões e decisões de arquitetura de dados.
    """)
    
    if st.button("🛠️ Compilar via pdflatex"):
        st.info("Processando arquivos .tex e gerando PDF auto-contido...")
        st.success("PDF gerado com sucesso!")
        st.download_button("📥 Download do Relatório Final (PDF)", data="Conteudo simulado", file_name="relatorio_projeto.pdf")