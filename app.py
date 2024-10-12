from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context


app = Flask(__name__)

# 加载数据集并预处理
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)

def process_query(query):
    query_vec = vectorizer.transform([query])
    query_reduced = svd.transform(query_vec)
    return query_reduced

def get_top_documents(query, top_n=5):
    query_reduced = process_query(query)
    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_similarities = similarities[top_indices]
    top_documents = [documents[i] for i in top_indices]
    return top_documents, top_similarities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    top_docs, top_sims = get_top_documents(query)
    return jsonify({'documents': top_docs, 'similarities': top_sims.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
