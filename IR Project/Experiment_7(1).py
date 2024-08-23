from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
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

def prepare_pairwise_training_data(indexed_data, queries, relevance_judgments):
    train_data = []
    for query_id, query_text in queries.items():
        relevant_docs = relevance_judgments.get(query_id, {})
        for relevant_doc_id in relevant_docs:
            relevant_doc_info = indexed_data.get(relevant_doc_id)
            for doc_id, doc_info in indexed_data.items():
                if doc_id != relevant_doc_id:
                    label = 1 if doc_id in relevant_docs else 0
                    train_data.append((query_text, relevant_doc_info.get("abstract", ""), doc_info.get("abstract", ""), label))
    return train_data

def generate_pairwise_features(train_data, tfidf_vectorizer):
    features = []
    for query, relevant_doc, irrelevant_doc, label in train_data:
        combined_relevant = query + " " + relevant_doc
        combined_irrelevant = query + " " + irrelevant_doc
        relevant_vector = tfidf_vectorizer.transform([combined_relevant])
        irrelevant_vector = tfidf_vectorizer.transform([combined_irrelevant])
        cosine_sim = cosine_similarity(relevant_vector, irrelevant_vector)[0][0]
        features.append((cosine_sim, label))
    return features

def prepare_listwise_training_data(indexed_data, queries, relevance_judgments):
    train_data = []
    for query_id, query_text in queries.items():
        relevant_docs = relevance_judgments.get(query_id, {})
        doc_infos = [(doc_id, doc_info) for doc_id, doc_info in indexed_data.items() if doc_id in relevant_docs]
        doc_infos.sort(key=lambda x: relevant_docs.get(x[0], 0), reverse=True)
        train_data.append((query_text, [doc_info[1].get("abstract", "") for doc_info in doc_infos]))
    return train_data

def generate_listwise_features(train_data, tfidf_vectorizer):
    features = []
    for query, doc_texts in train_data:
        combined_text = [query + " " + doc_text for doc_text in doc_texts]
        feature_vectors = tfidf_vectorizer.transform(combined_text)
        features.append(feature_vectors)
    return features

def train_regression_model(X_train, train_labels):
    regression_model = LinearRegression()
    regression_model.fit(X_train, train_labels)
    return regression_model

def evaluate_model(regression_model, indexed_data, queries, relevance_judgments):
    avg_map_score = 0
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
        avg_map_score += precision_at_k / len(relevant_docs) if relevant_docs else 0
    avg_map_score /= len(queries)
    return avg_map_score

if __name__ == "__main__":
    index_dir = "indexing"
    indexed_data = load_indexed_data(index_dir)
    queries = load_queries("nfcorpus/train.nontopic-titles.queries")
    relevance_judgments = load_relevance_judgments("nfcorpus/merged.qrel")

    pairwise_train_data = prepare_pairwise_training_data(indexed_data, queries, relevance_judgments)
    tfidf_vectorizer = TfidfVectorizer()
    pairwise_features = generate_pairwise_features(pairwise_train_data, tfidf_vectorizer)
    X_pairwise = [feat[0] for feat in pairwise_features]
    y_pairwise = [feat[1] for feat in pairwise_features]
    regression_model_pairwise = train_regression_model(X_pairwise, y_pairwise)

    listwise_train_data = prepare_listwise_training_data(indexed_data, queries, relevance_judgments)
    listwise_features = generate_listwise_features(listwise_train_data, tfidf_vectorizer)
    regression_model_listwise = train_regression_model(listwise_features, range(len(listwise_features)))

    avg_map_score_pairwise = evaluate_model(regression_model_pairwise, indexed_data, queries, relevance_judgments)
    avg_map_score_listwise = evaluate_model(regression_model_listwise, indexed_data, queries, relevance_judgments)

    print("Average MAP score for pairwise learning:", avg_map_score_pairwise)
    print("Average MAP score for listwise learning:", avg_map_score_listwise)

