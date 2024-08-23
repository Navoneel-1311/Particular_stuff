import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags

# Step 1: Preprocess the dataset

# Read queries
queries = {}
qtype = input("Enter type of query among titles, nontopic-titles, vid-titles, vid-desc: ")
with open("nfcorpus/test." + qtype + ".queries", "r", encoding='utf-8') as file:
    for line in file:
        query_id, query_text = line.strip().split("\t")
        queries[query_id] = query_text

# Read documents
documents = {}
with open("nfcorpus/raw/doc_dump.txt", "r", encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue
        doc_id, _, _, abstract = parts
        documents[doc_id] = abstract

# Read relevance judgments
relevance = {}
with open("nfcorpus/merged.qrel", "r") as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) != 4:
            continue
        query_id, _, doc_id, relevance_level = parts
        if query_id not in relevance:
            relevance[query_id] = {}
        relevance[query_id][doc_id] = int(relevance_level)

# Step 2: Implement Language Model (LM) ranking model

class LanguageModel:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer(smooth_idf=True, sublinear_tf=True, norm=None)

        # Calculate document-term matrix
        self.doc_term_matrix = csr_matrix(self.vectorizer.fit_transform(documents.values()))
        self.doc_term_matrix_tfidf = self.transformer.fit_transform(self.doc_term_matrix)

    def rank(self, query):
        query_vec = self.vectorizer.transform([query])
        query_vec_tfidf = self.transformer.transform(query_vec)

        similarity = cosine_similarity(query_vec_tfidf, self.doc_term_matrix_tfidf)
        rankings = np.argsort(similarity[0])[::-1]
        return rankings

# Step 3: Implement BM25 ranking model

class BM25:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

        # Calculate document-term matrix
        self.doc_term_matrix = csr_matrix(self.vectorizer.fit_transform(documents.values()))
        self.doc_term_matrix_tfidf = self.transformer.fit_transform(self.doc_term_matrix)

    def rank(self, query):
        query_vec = self.vectorizer.transform([query])
        query_vec_tfidf = self.transformer.transform(query_vec)

        k1 = 1.5
        b = 0.75
        avg_len_d = self.doc_term_matrix.sum(axis=1).mean()
        idf = self.transformer.idf_
        idf_diag = spdiags(idf, diags=0, m=self.doc_term_matrix_tfidf.shape[0], n=self.doc_term_matrix_tfidf.shape[0])

        bm25_scores = query_vec_tfidf.dot(self.doc_term_matrix_tfidf.T)
        doc_len = self.doc_term_matrix.sum(axis=1)
        doc_len_norm = (1 - b) + b * (doc_len / avg_len_d)

        bm25_scores = bm25_scores.multiply(idf_diag)
        bm25_scores = bm25_scores / (bm25_scores + k1 * doc_len_norm)

        row_indices, col_indices = bm25_scores.nonzero()
        bm25_scores_values = bm25_scores.data
        indices = np.argsort(-bm25_scores.data)
        rankings = row_indices[indices]
        return rankings

# Step 4: Evaluate the models

def evaluate_ranking(rankings, relevance, k=15):
    precision = []
    recall = []
    num_relevant = sum(rel in [2, 3] for doc_relevance in relevance.values() for rel in doc_relevance.values())
    for query_id, ranked_docs in rankings.items():
        ranked = ['MED-' + str(num) for num in ranked_docs]
        relevant_docs = [doc_id for doc_id, rel in relevance[query_id].items() if rel == 3 or rel == 2]
        num_retrieved = min(k, len(ranked_docs))
        num_relevant_retrieved = len(set(relevant_docs) & set(ranked[:k]))
        if num_relevant == 0:
            precision.append(0)
            recall.append(0)
        else:
            precision.append(num_relevant_retrieved / num_retrieved)
            recall.append(num_relevant_retrieved / num_relevant)
    return np.mean(precision), np.mean(recall)

# Step 5: Calculate NDCG score

def calculate_dcg(relevance_levels, k):
    dcg = 0
    for i in range(min(k, len(relevance_levels))):
        dcg += (2**relevance_levels[i] - 1) / np.log2(i + 2)
    return dcg

def calculate_ndcg(rankings, relevance, k=15):
    ndcg_scores = []
    for query_id, ranked_docs in rankings.items():
        relevant_docs = [rel for doc_id, rel in relevance[query_id].items()]
        dcg = calculate_dcg(relevant_docs, k)
        idcg = calculate_dcg(sorted(relevant_docs, reverse=True), k)
        if idcg == 0:
            ndcg_scores.append(0)
        else:
            ndcg_scores.append(dcg / idcg)
    return np.mean(ndcg_scores)

lm_model = LanguageModel(documents)
bm25_model = BM25(documents)

# Rank documents for each query
lm_rankings = {}
bm25_rankings = {}

for query_id, query_text in queries.items():
    lm_rankings[query_id] = lm_model.rank(query_text)
    bm25_rankings[query_id] = bm25_model.rank(query_text)

# Evaluate rankings
lm_precision, lm_recall = evaluate_ranking(lm_rankings, relevance)
bm25_precision, bm25_recall = evaluate_ranking(bm25_rankings, relevance)

print("Language Model:")
print("Precision:{:.10f}".format(lm_precision))
print("Recall:{:.10f}".format(lm_recall))

print("BM25 Model:")
print("Precision:{:.10f}".format(bm25_precision))
print("Recall:{:.10f}".format(bm25_recall))

lm_ndcg = calculate_ndcg(lm_rankings, relevance)
bm25_ndcg = calculate_ndcg(bm25_rankings, relevance)

print("NDCG Scores:")
print("Language Model:", lm_ndcg)
print("BM25 Model:", bm25_ndcg)
