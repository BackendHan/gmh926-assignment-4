# lsa.py

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context


# 加载所有数据
def load_documents():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data
    return documents

# 创建TF-IDF矩阵
def create_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(documents)
    return X, vectorizer

# 应用SVD进行降维
def apply_svd(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    return svd, X_reduced

# 处理用户查询
def process_query(query, vectorizer, svd):
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)
    return query_reduced

# 计算余弦相似度并检索前N个文档
def get_top_documents(query, documents, X_reduced, vectorizer, svd, top_n=5):
    query_reduced = process_query(query, vectorizer, svd)
    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_similarities = similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]
    return top_documents, top_similarities

# 示例用法
if __name__ == "__main__":
    # 加载文档
    documents = load_documents()

    # 创建TF-IDF矩阵
    X, vectorizer = create_tfidf_matrix(documents)

    # 应用SVD降维
    svd, X_reduced = apply_svd(X)

    # 示例查询
    query = "machine learning"

    # 获取最相关的文档
    top_docs, top_sims = get_top_documents(query, documents, X_reduced, vectorizer, svd)

    # 输出结果
    for idx, (doc, sim) in enumerate(zip(top_docs, top_sims)):
        print(f"文档 {idx+1} (相似度: {sim:.4f}):\n{doc[:200]}...\n")
