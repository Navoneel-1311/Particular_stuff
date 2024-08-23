import numpy as np
import math
from collections import Counter

def load_queries(file_path):
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            query_id, query_text = line.strip().split('\t')
            queries[query_id] = query_text
    return queries

def load_documents(file_path):
    documents = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            doc_id, _, _, abstract = line.strip().split('\t')
            documents[doc_id] = abstract
    return documents

def load_relevance(file_path):
    relevance = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            query_id, _, doc_id, relevance_level = parts
            if relevance_level != '0':
                if query_id not in relevance:
                    relevance[query_id] = {}
                relevance[query_id][doc_id] = int(relevance_level)
    return relevance

def compute_term_frequency(text):
    if text is None:
        return {}
    word_counts = Counter(text.split())
    term_frequencies = {word: count / len(text.split()) for word, count in word_counts.items()}
    return term_frequencies

def compute_idf_values(documents):
    idf_values = {}
    num_documents = len(documents)
    for document in documents.values():
        words = set(document.split())
        for word in words:
            if word not in idf_values:
                document_frequency = sum(1 for doc in documents.values() if word in doc)
                idf_values[word] = np.log((num_documents + 1) / (document_frequency + 1))
    return idf_values

def compute_tf_idf(document, idf_values):
    document_tf = compute_term_frequency(document)
    document_tf_idf = {word: tf * idf_values[word] for word, tf in document_tf.items()}
    return document_tf_idf

def retrieve_relevant_documents(query_id, relevance):
    relevant_docs = []
    if query_id in relevance:
        relevant_docs = list(relevance[query_id].keys())
    return relevant_docs

def compute_document_vectors(documents, idf_values):
    document_vectors = {}
    for doc_id, document in documents.items():
        document_tf_idf = compute_tf_idf(document, idf_values)
        document_vector = np.zeros(len(idf_values))
        for word, tfidf in document_tf_idf.items():
            if word in idf_values:
                document_vector[list(idf_values.keys()).index(word)] = tfidf
        document_vectors[doc_id] = document_vector
    return document_vectors

def update_query_vector(query_id, query_vectors, documents, idf_values, relevant_docs, irrelevant_docs, alpha=1, beta=0.75, gamma=0.15):
    query_vector = query_vectors[query_id]
    relevant_vectors = np.array([document_vectors[doc_id] for doc_id in relevant_docs])
    irrelevant_vectors = np.array([document_vectors[doc_id] for doc_id in irrelevant_docs])
    
    relevant_mean = np.mean(relevant_vectors, axis=0) if relevant_vectors.size > 0 else np.zeros(len(idf_values))
    irrelevant_mean = np.mean(irrelevant_vectors, axis=0) if irrelevant_vectors.size > 0 else np.zeros(len(idf_values))
    
    updated_query_vector = alpha * query_vector + beta * relevant_mean - gamma * irrelevant_mean
    norm = np.linalg.norm(updated_query_vector)
    if norm != 0:
        query_vectors[query_id] = updated_query_vector / norm

def expand_queries(queries_file, documents_file, relevance_file, alpha=1, beta=0.75, gamma=0.15, max_iterations=5):
    queries = load_queries(queries_file)
    documents = load_documents(documents_file)
    relevance = load_relevance(relevance_file)
    idf_values = compute_idf_values(documents)
    document_vectors = compute_document_vectors(documents, idf_values)

    for query_id in queries:
        query_vector = np.zeros(len(idf_values))
        relevant_docs = retrieve_relevant_documents(query_id, relevance)
        irrelevant_docs = list(set(documents.keys()) - set(relevant_docs))

        iteration = 0
        while iteration < max_iterations:
            update_query_vector(query_id, query_vectors, documents, idf_values, relevant_docs, irrelevant_docs, alpha, beta, gamma)
            iteration += 1

        print(f"Expanded Query ID: {query_id}")
        print("Expanded Query Vector:", query_vector)
        if np.any(query_vector):
            print("Non-zero elements exist in the query vector.")
        else:
            print("Query vector consists entirely of zeros.")
        break

if __name__ == "__main__":
    qtype = input("Enter type of query among titles, nontopic-titles, vid-titles, vid-desc: ")
    queries_file = 'nfcorpus/test.' + qtype + '.queries'
    documents_file = 'nfcorpus/raw/doc_dump.txt'
    relevance_file = 'nfcorpus/merged.qrel'
    corpus_file = 'corpus.txt'
    expanded_queries = expand_queries(queries_file, documents_file, relevance_file)
