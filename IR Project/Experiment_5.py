import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index
from collections import Counter

def load_entities(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        entities = [line.strip() for line in file]
    return entities

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

def load_inverted_indexes(filename):
    inverted_indexes = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            term = parts[0]
            postings = parts[1:]
            postings_dict = {}
            for posting in postings:
                doc_id, freq = posting.split(':')
                postings_dict[doc_id] = int(freq)
            inverted_indexes[term] = postings_dict
    return inverted_indexes

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

def create_document_vectors(entities, inverted_indexes):
    document_vectors = {}
    for term, postings in inverted_indexes.items():
        for doc_id, freq in postings.items():
            if doc_id not in document_vectors:
                document_vectors[doc_id] = np.zeros(len(entities))
            for entity in entities:
                if entity == term:
                    document_vectors[doc_id][entities.index(entity)] += 1
    return document_vectors

def calculate_similarity(query_vector, document_vector):
    return cosine_similarity([query_vector], [document_vector])[0][0]

def rank_documents(query_vector, document_vectors):
    rankings = {}
    for doc_id, document_vector in document_vectors.items():
        similarity = calculate_similarity(query_vector, document_vector)
        if similarity > 0:
            rankings[doc_id] = similarity
    return sorted(rankings.items(), key=lambda x: x[1], reverse=True)

def output_results(query_id, ranked_documents, output_file):
    with open(output_file, 'a', encoding='utf-8') as file:
        for rank, (doc_id, similarity) in enumerate(ranked_documents, start=1):
            file.write(f"{query_id}\t{doc_id}\t{similarity}\t{rank}\n")

def evaluate_model(model_scores, relevance_judgments):
    precision_scores = {}
    map_score = 0

    for query_id, doc_scores in model_scores.items():
        sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)

        relevant_docs = relevance_judgments.get(query_id, {})

        precision_at_15 = 0
        for rank, doc_id in enumerate(sorted_doc_ids[:15]):
            if doc_id in relevant_docs:
                precision_at_15 += 1 / (rank + 1)

        precision_scores[query_id] = precision_at_15
        map_score += precision_at_15 / len(relevant_docs) if relevant_docs else 0

    map_score = map_score / len(precision_scores)

    ndcg_score = calculate_ndcg(model_scores, relevance_judgments)

    return map_score, ndcg_score

def calculate_dcg(scores, k):
    dcg = 0
    for i in range(1, min(k, len(scores)) + 1):
        dcg += (2 ** scores[i - 1] - 1) / math.log2(i + 1)
    return dcg

def calculate_ndcg(model_scores, relevance_judgments, k=15):
    ndcg_scores = {}
    for query_id, doc_scores in model_scores.items():
        sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        
        relevant_docs = relevance_judgments.get(query_id, {})
        
        ranked_relevance = [relevant_docs.get(doc_id, 0) for doc_id in sorted_doc_ids[:k]]
        
        ideal_sorted_docs = sorted(relevant_docs, key=relevant_docs.get, reverse=True)
        ideal_relevance = [relevant_docs[doc_id] for doc_id in ideal_sorted_docs[:k]]
    
        dcg = calculate_dcg(ranked_relevance, k)
        ideal_dcg = calculate_dcg(ideal_relevance, k)
        
        if ideal_dcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / ideal_dcg
        ndcg_scores[query_id] = ndcg
    
    avg_ndcg = sum(ndcg_scores.values()) / len(ndcg_scores)
    return avg_ndcg

if __name__ == "__main__":
    index_dir = "indexing"
    entities = load_entities("entities.txt")
    inverted_indexes = load_inverted_indexes("inverted_index.tsv")
    queries = load_queries("nfcorpus/dev.nontopic-titles.queries")
    relevance_judgments = load_relevance_judgments("nfcorpus/merged.qrel")

    document_vectors = create_document_vectors(entities, inverted_indexes)

    model_scores = {}
    for query_id, query in queries.items():
        query_vector = np.zeros(len(entities))
        for term in query.split():
            if term in entities:
                query_vector[entities.index(term)] += 1

        ranked_documents = rank_documents(query_vector, document_vectors)
        model_scores[query_id] = {doc_id: similarity for doc_id, similarity in ranked_documents}

        output_results(query_id, ranked_documents[:15], "output.txt")

    map_score, ndcg_score = evaluate_model(model_scores, relevance_judgments)
    print("MAP Score:", map_score)
    print("NDCG Score:", ndcg_score)

