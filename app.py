from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from auxiliar import *
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
from dotenv import load_dotenv
import toml

config = toml.load('config.toml')

# Definir a variável de ambiente DATA_PATH
os.environ['DATA_PATH'] = config['environment']['DATA_PATH']


data_path = os.getenv('DATA_PATH')

config_yaml_path = os.path.join(data_path, 'config.yaml')
config_data = load_config(config_yaml_path)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout()
    st.write(f'Bem-vindo/a  *{st.session_state["name"]}*! ')
    @st.cache_resource
    def cluster_model():
        # 3. Converter textos em vetores de características usando TF-IDF
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        X = vectorizer.fit_transform(df['Info_Completa'])

        # 4. Aplicar K-means para clusterização
        true_k = 3  # Suponha que queremos 3 clusters
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)

        return model


    @st.cache_data
    def get_data(path):
        cols = [
            "Código", "Data Admissão", "Data Saída",
            "Empresa", "Área", "Função", "Gestor", "Tempo de empresa",
            "Motivo do desligamento", "Selecione o principal fator que motivou o seu desligamento:",
            "Informe o motivo mencionado no item anterior",
            "Na sua opinião, quais aspectos importantes a empresa deveria desenvolver?",
            "Espaço livre para sugestões, elogios, críticas e afins"
        ]
        df = pd.read_excel(path, names=cols)
        df.columns = df.columns.str.replace(' ', '_')
        df['Motivo_do_desligamento'] = df['Motivo_do_desligamento'].replace({
            'Inciativa do colaborador': 'Iniciativa do colaborador'
        })
        df['Motivo_do_desligamento'] = df['Motivo_do_desligamento'].replace({
            'Iniciativa da empresa': 'Iniciativa de empresa'
        })

        df['Info_Completa'] = df[["Selecione_o_principal_fator_que_motivou_o_seu_desligamento:",
                                  "Informe_o_motivo_mencionado_no_item_anterior",
                                  "Na_sua_opinião,_quais_aspectos_importantes_a_empresa_deveria_desenvolver?",
                                  "Espaço_livre_para_sugestões,_elogios,_críticas_e_afins"]].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1)
        return df


    @st.cache_resource
    def analisar_sentimento(texto):
        texto_traduzido = traduzir_texto(texto)
        score = analyzer.polarity_scores(texto_traduzido)
        return score['compound']


    @st.cache_resource
    def get_stop_words():
        nltk.download('stopwords')
        stopwords = nltk.corpus.stopwords.words('portuguese')
        stopwords.extend(['de', 'outro', 'que', 'o', 'se', 'um', 'para', 'não', 'e', 'por', 'da',
                          'mais', 'é', 'em', 'outra', 'ter', 'na', 'uma', 'mas', 'como', 'tem', 'estava'
                             , 'pra', 'outras', 'herbarium', 'sobre', 'forma', 'tudo', 'pois'])

        return stopwords


    st.title('EDA')

    path = f"{data_path}/Entrevista de Desligamento 2023_2024.xlsx"
    df = get_data(path)
    df_orig = df.copy()

    motivos_contagem = df['Motivo_do_desligamento'].value_counts()

    # Gráfico de barras dos motivos de desligamento
    plt.figure(figsize=(10, 6))
    sns.barplot(x=motivos_contagem.values, y=motivos_contagem.index)
    plt.title('Principais Motivos de Desligamento')
    plt.xlabel('Frequência')
    plt.ylabel('Motivo')

    st.pyplot(plt)

    st.write("Resumo geral:")

    stopwords = get_stop_words()
    option = st.selectbox(
        'Selecione um motivo de desligamento:',
        df["Motivo_do_desligamento"].unique(), key="1")

    col = 'Info_Completa'
    text = ' '.join(df[col].astype(str).tolist())
    # Gere a nuvem de palavras
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
    # Exibir a nuvem de palavras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Desliga os eixos

    st.pyplot(plt)

    st.header("______________________________________________________")

    st.write("Principal fator por área:")

    st.button("Escolher Área", type="primary")

    if st.button("Todas"):
        try:
            col = 'Selecione_o_principal_fator_que_motivou_o_seu_desligamento:'
            df = df[df["Motivo_do_desligamento"] == option]
            text = ' '.join(df[col].astype(str).tolist())
            # Gere a nuvem de palavras
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
            # Exibir a nuvem de palavras
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Desliga os eixos

            st.pyplot(plt)
        except:
            st.write("Sem palavras para gerar grafico!")
    else:
        option_area = st.selectbox(
            'Selecione uma Área:',
            df['Área'].unique(), key="2")
        try:
            col = 'Selecione_o_principal_fator_que_motivou_o_seu_desligamento:'
            df = df[(df["Motivo_do_desligamento"] == option) & (df["Área"] == option_area)]
            text = ' '.join(df[col].astype(str).tolist())
            # Gere a nuvem de palavras
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)
            # Exibir a nuvem de palavras
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')  # Desliga os eixos

            st.pyplot(plt)
        except:
            st.write("Sem palavras para gerar grafico!")

    st.header("______________________________________________________")
    ############## Clusterização

    try:
        model = cluster_model()

        # Adicionar os clusters aos textos originais para visualização
        df['Cluster'] = model.labels_

        # 6. Visualizar usando PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X.toarray())
        df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
        df_pca['Cluster'] = model.labels_

        # Certifique-se de que a coluna 'Cluster' seja tratada como categoria
        df_pca['Cluster'] = df_pca['Cluster'].astype(str)

        # Usar Plotly para visualização
        fig = px.scatter(
            df_pca,
            x='Principal Component 1',
            y='Principal Component 2',
            color='Cluster',
            title="Clusterização de funcionários",
            labels={"Principal Component 1": "Componente Principal 1",
                    "Principal Component 2": "Componente Principal 2"}
        )

        # Mostrar a legenda dos clusters no gráfico
        fig.update_layout(legend_title_text='Clusters')

        st.plotly_chart(fig)

        # Visualizar os clusters
        st.markdown("# Top termos por cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(true_k):
            st.markdown(f"#### Cluster {i}:")
            terms_list = []
            for ind in order_centroids[i, :10]:
                terms_list.append(terms[ind])
            st.markdown(f" {', '.join(terms_list)}")

        df.reset_index(inplace=True, drop=True)

        df = df[["Cluster", "Área", "Tempo_de_empresa",
                 "Código",
                 "Data_Admissão",
                 "Data_Saída",
                 "Empresa",
                 "Função",
                 "Gestor",
                 "Motivo_do_desligamento",
                 "Selecione_o_principal_fator_que_motivou_o_seu_desligamento:",
                 "Informe_o_motivo_mencionado_no_item_anterior",
                 "Na_sua_opinião,_quais_aspectos_importantes_a_empresa_deveria_desenvolver?",
                 "Espaço_livre_para_sugestões,_elogios,_críticas_e_afins",
                 "Info_Completa"

                 ]]

        cluster_group = st.multiselect('Cluster', options=df['Cluster'].unique())

        filtered_df = df
        if cluster_group:
            filtered_df = filtered_df[filtered_df['Cluster'].isin(cluster_group)]

        st.dataframe(df)
    except:
        st.write("Não foi possível gerar clusters!")

    st.header("______________________________________________________")
    st.title('Distribuição de sentimentos das sugestões de melhorias')

    filtered_df2 = df_orig

    filtered_df2['campo_livre'] = filtered_df2[["Informe_o_motivo_mencionado_no_item_anterior",
                                                "Na_sua_opinião,_quais_aspectos_importantes_a_empresa_deveria_desenvolver?",
                                                "Espaço_livre_para_sugestões,_elogios,_críticas_e_afins"]].apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1)

    tempo_empresa = st.multiselect('Tempo de empresa', options=filtered_df2['Tempo_de_empresa'].unique())
    area = st.multiselect('Área', options=filtered_df2['Área'].unique())

    if tempo_empresa:
        filtered_df2 = filtered_df2[filtered_df2['Tempo_de_empresa'].isin(tempo_empresa)]
    if area:
        filtered_df2 = filtered_df2[filtered_df2['Área'].isin(area)]

    filtro_data = st.slider(
        'Data admissão',
        min_value=filtered_df2['Data_Admissão'].min().to_pydatetime(),
        max_value=filtered_df2['Data_Admissão'].max().to_pydatetime(),
        value=(
        filtered_df2['Data_Admissão'].min().to_pydatetime(), filtered_df2['Data_Admissão'].max().to_pydatetime()),
        format="YYYY-MM-DD"
    )

    # Filtrando o DataFrame com base no intervalo selecionado
    filtered_df2 = filtered_df2[
        (filtered_df2['Data_Admissão'] >= filtro_data[0]) & (filtered_df2['Data_Admissão'] <= filtro_data[1])]

    # Supondo que df já está carregado e possui a coluna especificada
    # Aplicar a análise de sentimento à coluna especificada
    filtered_df2['Sentimento'] = filtered_df2['campo_livre'].apply(analisar_sentimento)

    # Visualização dos sentimentos

    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_df2['Sentimento'], bins=20, kde=True)
    plt.title('Análise de sentimentos')
    plt.xlabel('Sentimento')
    plt.ylabel('Frequência')

    # Mostrar o gráfico no Streamlit
    st.pyplot(plt.gcf())
    filtered_df2.reset_index(inplace=True, drop=True)
    linha_selecionada = st.selectbox("Selecione a Linha", filtered_df2.index)

    st.dataframe(filtered_df2[['Sentimento', 'campo_livre']])

    linha = filtered_df2.loc[linha_selecionada, 'campo_livre']
    st.text_area(f"Conteúdo Completo da Linha {linha_selecionada}, Coluna 'campo_livre'", linha, height=200)


elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

