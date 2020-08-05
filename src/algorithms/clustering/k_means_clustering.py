from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def run_algo(articles):
    tf_idf_vectorizor = TfidfVectorizer(stop_words='english',  # tokenizer = tokenize_and_stem,
                                        max_features=20000)
    tf_idf = tf_idf_vectorizor.fit_transform(articles)
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()

    sklearn_pca = PCA(n_components=2)
    Y_sklearn = sklearn_pca.fit_transform(tf_idf_array)

    #elbow method to determine clusters
    #number_clusters = range(1, 15)
    #kmeans = [KMeans(n_clusters=i, max_iter=600) for i in number_clusters]
    #score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]
    #plt.plot(number_clusters, score)
    #plt.xlabel('Number of Clusters')
    #plt.ylabel('Score')
    #plt.title('Elbow Method')
    #plt.show()

    kmeans = KMeans(n_clusters=6, max_iter=600, algorithm='auto')
    prediction = kmeans.predict(Y_sklearn)
    return prediction
