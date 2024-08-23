from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from whoosh import index

def load_indexed_data(index_dir):
    indexed_data = {}
    ix = index.open_dir(index_dir)
    with ix.searcher() as searcher:
        results = searcher.all_stored_fields()
        for result in results:
            doc_id = result["id"]
            indexed_data[doc_id] = {
                "url": result.get("url", None),  
                "title": result.get("title", None),  
                "abstract": result.get("abstract", None)  
            }
    return indexed_data

def load_queries(query_filename):
    queries = {}
    with open(query_filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                query_id, query_text = parts
                queries[query_id.strip()] = query_text.strip()
    return queries

def load_relevance_judgments(file_path):
    relevance_judgments = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            query_id, _, doc_id, relevance = parts
            if query_id not in relevance_judgments:
                relevance_judgments[query_id] = {}
            relevance_judgments[query_id][doc_id] = int(relevance)
    return relevance_judgments

def prepare_training_data(indexed_data, queries, relevance_judgments):
    train_data = []
    train_labels = []
    for query_id, query_text in queries.items():
        for doc_id, doc_info in indexed_data.items():
            relevance = relevance_judgments.get(query_id, {}).get(doc_id, 0)
            train_data.append((query_text, doc_info["abstract"]))  
            train_labels.append(relevance)  
    return train_data, train_labels

def generate_features(train_data):
    combined_text = [query + " " + doc for query, doc in train_data if query is not None and doc is not None]
    tfidf_vectorizer = TfidfVectorizer()
    X_train = tfidf_vectorizer.fit_transform(combined_text)
    return X_train

def train_regression_model(X_train, train_labels):
    regression_model = LinearRegression()
    regression_model.fit(X_train, train_labels)
    return regression_model

def evaluate_model(regression_model, indexed_data, queries, relevance_judgments):
    map_score = 0
    for query_id, query_text in queries.items():
        relevant_docs = relevance_judgments.get(query_id, {})
        scores = []
        for doc_id, doc_info in indexed_data.items():
            features = tfidf_vectorizer.transform([query_text + " " + doc_info["abstract"]])
            score = regression_model.predict(features)
            scores.append((doc_id, score))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        precision_at_k = 0
        for rank, (doc_id, _) in enumerate(sorted_scores[:15], start=1):
            if doc_id in relevant_docs:
                precision_at_k += 1 / rank
        map_score += precision_at_k / len(relevant_docs) if relevant_docs else 0
    avg_map_score = map_score / len(queries)
    return avg_map_score

if __name__ == "__main__":
    index_dir = "indexing"
    indexed_data = load_indexed_data(index_dir)
    queries = load_queries("nfcorpus/train.nontopic-titles.queries")
    relevance_judgments = load_relevance_judgments("nfcorpus/merged.qrel")

    train_data, train_labels = prepare_training_data(indexed_data, queries, relevance_judgments)

    X_train = generate_features(train_data)

    regression_model = train_regression_model(X_train, train_labels)

    avg_map_score = evaluate_model(regression_model, indexed_data, queries, relevance_judgments)
    print("Average MAP score:", avg_map_score)

