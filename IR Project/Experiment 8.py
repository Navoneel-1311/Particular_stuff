import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from whoosh import index
from collections import Counter
import os
from nltk.corpus import wordnet as wn
from itertools import chain

#Step 1: Load the dataset and queries
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

def expand_query_with_synonyms(query):
    expanded_query = []
    for term in query.split():
        synonyms = set(chain.from_iterable([word.lemma_names() for word in wn.synsets(term)]))
        expanded_query.extend(synonyms)
    return ' '.join(expanded_query)

def load_queries(query_filename, expand_query = True):
    queries = {}
    with open(query_filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                query_id, query_text = parts
                if expand_query:
                    expanded_query_text = expand_query_with_synonyms(query_text)
                    queries[query_id.strip()] = expanded_query_text.strip()
                else:
                    queries[query_id.strip()] = query_text.strip()
    return queries

#Step 2: Prepare data for training, development and testing and load the corpus
def prepare_data(document):
    document_text = {doc_id: doc_info["abstract"] for doc_id, doc_info in document.items()}
    return document_text

def load_corpus(corpus_file):
  corpus = []
  with open(corpus_file, 'r', encoding='utf-8') as file:
    for line in file:
      corpus.append(line.strip())
  return corpus

#Step 3: Define the 3 models and its calculations
def calculate_term_frequency(text):
  if text is None:
    return {}
  word_counts = Counter(text.split())
  term_frequencies = {word: count / len(text.split()) for word, count in word_counts.items()}
  return term_frequencies

def nnn_model(queries, documents, rel_docs, corpus):
  result = {}
  for query in queries.keys():
    intermediate = {}
    query_tf = calculate_term_frequency(queries[query])
    for doc_id in rel_docs[query]:
      document_tf = calculate_term_frequency(documents[doc_id])
      dot_product = sum(query_tf[word] * document_tf.get(word, 0) for word in query_tf)
      intermediate[doc_id] = dot_product
    result[query] = intermediate
    print('Query: ',query,' completed')
  print('NNN completed.')
  return result

def ntn_model(queries, documents, rel_docs, corpus):
  result = {}
  for query in queries.keys():
    intermediate = {}
    query_tf = calculate_term_frequency(queries[query])
    for doc_id in rel_docs[query]:
      document_tf = calculate_term_frequency(documents[doc_id])
      document_tf_idf = calculate_tf_idf(documents[query], corpus)
      dot_product = sum(query_tf[word] * document_tf.get(word, 0) for word in query_tf)
      intermediate[doc_id] = dot_product
    result[query] = intermediate
    print('Query: ',query,' completed')
  print('NTN completed.')
  return result

def calculate_idf(word, corpus):
  num_documents = len(corpus)
  document_frequency = sum(1 for doc in corpus if word in doc)
  idf = np.log((num_documents + 1) / (document_frequency + 1))
  return idf

def calculate_tf_idf(document, corpus):
  document_tf = calculate_term_frequency(document)
  document_tf_idf = {word: tf * calculate_idf(word, corpus) for word, tf in document_tf.items()}
  return document_tf_idf

def custom_cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0  # Handle division by zero
    return dot_product / (magnitude1 * magnitude2)

def ntc_model(queries, documents, rel_docs, corpus):
    result = {}
    for query in queries.keys():
        intermediate = {}
        query_tf = calculate_term_frequency(queries[query])
        for doc_id in rel_docs[query]:
            document_tf = calculate_term_frequency(documents[doc_id])
            dot_product = sum(query_tf[word] * document_tf.get(word, 0) for word in query_tf)
            intermediate[doc_id] = dot_product
        result[query] = intermediate
    
    for query, doc_scores in result.items():
        query_vector = [doc_scores[doc_id] for doc_id in doc_scores]
        for doc_id in doc_scores:
            document_vector = [doc_scores[doc_id] for _ in range(len(query_vector))]
            cosine_sim = custom_cosine_similarity(query_vector, document_vector)
            result[query][doc_id] = cosine_sim
        print('Query: ', query, ' completed')
    print('NTC completed')
    return result

# Step 4: Train the models
def train_models(query_vectors_train, document_vectors_train, rel_docs_train, corpus):
    nnn_scores_train = nnn_model(query_vectors_train, document_vectors_train, rel_docs_train, corpus)
    ntn_scores_train = ntn_model(query_vectors_train, document_vectors_train, rel_docs_train, corpus)
    ntc_scores_train = ntc_model(query_vectors_train, document_vectors_train, rel_docs_train, corpus)
    return nnn_scores_train, ntn_scores_train, ntc_scores_train

#Step 5: Evaluate the models on development/testing data
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

def evaluate_model(model_scores, relevance_judgments):
  precision_scores = {}
  map_score = 0

  for query_id, doc_scores in model_scores.items():
    # Sort documents by their retrieval scores in descending order
    sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)

    # Get relevant documents for this query
    relevant_docs = relevance_judgments.get(query_id, {})

    # Calculate precision at different ranks (top 1, top 5, top 15, etc.), here k = 15
    precision_at_15 = 0
    for rank, doc_id in enumerate(sorted_doc_ids[:15]):
      if doc_id in relevant_docs:
        precision_at_15 += 1 / (rank + 1)

    precision_scores[query_id] = precision_at_15
    map_score += precision_at_15 / len(relevant_docs) if relevant_docs else 0

  # Calculate mean average precision (MAP)
  map_score = map_score / len(precision_scores)
  return map_score

#Step 6: Calculate NDCG score for the models
def calculate_dcg(scores, k):
    dcg = 0
    for i in range(1, min(k, len(scores)) + 1):
        dcg += (2 ** scores[i - 1] - 1) / math.log2(i + 1)
    return dcg

def calculate_ndcg(model_scores, relevance_judgments, k = 15):
    ndcg_scores = {}
    for query_id, doc_scores in model_scores.items():
        # Sort documents by their retrieval scores in descending order
        sorted_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)
        
        # Get relevance judgments for this query
        relevant_docs = relevance_judgments.get(query_id, {})
        
        # Compute relevance scores for the top-ranked documents
        ranked_relevance = [relevant_docs.get(doc_id, 0) for doc_id in sorted_doc_ids[:k]]
        
        # Compute ideal relevance scores
        ideal_sorted_docs = sorted(relevant_docs, key=relevant_docs.get, reverse=True)
        ideal_relevance = [relevant_docs[doc_id] for doc_id in ideal_sorted_docs[:k]]
        
        # Calculate DCG and ideal DCG
        dcg = calculate_dcg(ranked_relevance, k)
        ideal_dcg = calculate_dcg(ideal_relevance, k)
        
        # Compute NDCG
        if ideal_dcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / ideal_dcg
        ndcg_scores[query_id] = ndcg
    
    # Calculate average NDCG across all queries
    avg_ndcg = sum(ndcg_scores.values()) / len(ndcg_scores)
    return avg_ndcg

if __name__ == "__main__":
    # Step 1: Load indexed data
    index_dir = "indexing"
    document = prepare_data(load_indexed_data(index_dir))
    corpus = load_corpus("corpus.txt")
    
    # Step 2: Prepare data for training and testing
    qtype = input("Enter type of query among titles, nontopic-titles, vid-titles, vid-desc: ")
    queries_train = load_queries("nfcorpus/train." + qtype + ".queries")
    queries_dev = load_queries("nfcorpus/dev." + qtype + ".queries")
    queries_test = load_queries("nfcorpus/test." + qtype + ".queries")
    relevance_judgments = load_relevance_judgments("nfcorpus/merged.qrel")

    #Step 3: Find out the relevant documents using qrel file
    rel_docs_train, rel_docs_dev, rel_docs_test = {}, {}, {}
    '''for query in queries_train.keys():
        l = list(relevance_judgments[query].keys())
        rel_docs_train[query] = l

    for query in queries_dev.keys():
        l = list(relevance_judgments[query].keys())
        rel_docs_dev[query] = l'''

    for query in queries_test.keys():
        l = list(relevance_judgments[query].keys())
        rel_docs_test[query] = l

    # Step 4: Train the models
    #nnn_scores_train, ntn_scores_train, ntc_scores_train = train_models(queries_train, document, rel_docs_train, corpus)
    #nnn_scores_dev, ntn_scores_dev, ntc_scores_dev = train_models(queries_dev, document, rel_docs_dev, corpus)
    nnn_scores_test, ntn_scores_test, ntc_scores_test = train_models(queries_test, document, rel_docs_test, corpus)

    # Step 5: Evaluate the models on all data
    '''nnn_map_score_train = evaluate_model(nnn_scores_train, relevance_judgments)
    ntn_map_score_train = evaluate_model(ntn_scores_train, relevance_judgments)
    ntc_map_score_train = evaluate_model(ntc_scores_train, relevance_judgments)
    
    nnn_map_score_dev = evaluate_model(nnn_scores_dev, relevance_judgments)
    ntn_map_score_dev = evaluate_model(ntn_scores_dev, relevance_judgments)
    ntc_map_score_dev = evaluate_model(ntc_scores_dev, relevance_judgments)'''

    nnn_map_score_test = evaluate_model(nnn_scores_test, relevance_judgments)
    ntn_map_score_test = evaluate_model(ntn_scores_test, relevance_judgments)
    ntc_map_score_test = evaluate_model(ntc_scores_test, relevance_judgments)

    '''print("MAP score for nnn model on training data:", nnn_map_score_train)
    print("MAP score for ntn model on training data:", ntn_map_score_train)
    print("MAP score for ntc model on training data:", ntc_map_score_train)

    print("MAP score for nnn model on development data:", nnn_map_score_dev)
    print("MAP score for ntn model on development data:", ntn_map_score_dev)
    print("MAP score for ntc model on development data:", ntc_map_score_dev)'''

    print("MAP score for nnn model on testing data:", nnn_map_score_test)
    print("MAP score for ntn model on testing data:", ntn_map_score_test)
    print("MAP score for ntc model on testing data:", ntc_map_score_test)

    #Step 6: Calculate NDCG score for all models
    k = 15 # We are taking top 15 results
    
    '''nnn_ndcg_train = calculate_ndcg(nnn_scores_train, relevance_judgments, k)
    ntn_ndcg_train = calculate_ndcg(ntn_scores_train, relevance_judgments, k)
    ntc_ndcg_train = calculate_ndcg(ntc_scores_train, relevance_judgments, k)

    nnn_ndcg_dev = calculate_ndcg(nnn_scores_dev, relevance_judgments, k)
    ntn_ndcg_dev = calculate_ndcg(ntn_scores_dev, relevance_judgments, k)
    ntc_ndcg_dev = calculate_ndcg(ntc_scores_dev, relevance_judgments, k)'''

    nnn_ndcg_test = calculate_ndcg(nnn_scores_test, relevance_judgments, k)
    ntn_ndcg_test = calculate_ndcg(ntn_scores_test, relevance_judgments, k)
    ntc_ndcg_test = calculate_ndcg(ntc_scores_test, relevance_judgments, k)

    '''print("NDCG score for nnn model on training data:", nnn_ndcg_train)
    print("NDCG score for ntn model on training data:", ntn_ndcg_train)
    print("NDCG score for ntc model on training data:", ntc_ndcg_train)

    print("NDCG score for nnn model on development data:", nnn_ndcg_dev)
    print("NDCG score for ntn model on development data:", ntn_ndcg_dev)
    print("NDCG score for ntc model on development data:", ntc_ndcg_dev)'''

    print("NDCG score for nnn model on testing data:", nnn_ndcg_test)
    print("NDCG score for ntn model on testing data:", ntn_ndcg_test)
    print("NDCG score for ntc model on testing data:", ntc_ndcg_test)
