from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import nltk

def clustering(df):
    """

    :param df:
    :return:
    """

    try:
        # 3. Converter textos em vetores de características usando TF-IDF
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        X = vectorizer.fit_transform(df['Info_Completa'])

        # 4. Aplicar K-means para clusterização
        true_k = 3  # Suponha que queremos 3 clusters
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)

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
            labels={"Principal Component 1": "Componente Principal 1", "Principal Component 2": "Componente Principal 2"}
        )

        # Mostrar a legenda dos clusters no gráfico
        fig.update_layout(legend_title_text='Clusters')

        st.plotly_chart(fig)

        st.write("Top termos por cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(true_k):
            st.write(f"Cluster {i}:")
            terms_list = []
            for ind in order_centroids[i, :10]:
                terms_list.append(terms[ind])
            st.write(", ".join(terms_list))

    except Exception as e:
        print("Sem dados suficientes para clusterizar!")

# Baixar o léxico do VADER, se ainda não foi baixado
nltk.download('vader_lexicon')
# Criar instâncias do analisador de sentimentos e do tradutor
analyzer = SentimentIntensityAnalyzer()
translator = Translator()
def traduzir_texto(texto):
    try:
        traducao = translator.translate(texto, src='pt', dest='en')
        return traducao.text
    except Exception as e:
        print(f"Erro na tradução: {e}")
        return texto

# Função para obter o sentimento
