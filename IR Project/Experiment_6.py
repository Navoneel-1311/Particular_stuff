import csv
from collections import defaultdict

def load_queries(filename):
    queries = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            query_id, query = line.strip().split('\t')
            queries[query_id] = query
    return queries

def load_knowledge_graph(filename):
    knowledge_graph = defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            subject, relation, obj = row
            knowledge_graph[subject].append((relation, obj))
    return knowledge_graph

def extract_entities(query, knowledge_graph):
    entities = set()
    for word in query.split():
        if word in knowledge_graph:
            entities.add(word)
    return entities

def expand_query(entities, knowledge_graph, original_query):
    expanded_query = set(entities)
    for entity in entities:
        if entity in knowledge_graph:
            for relation, related_entity in knowledge_graph[entity]:
                expanded_query.add(related_entity)
    if len(expanded_query) > 0:
        return ' '.join(expanded_query)
    else:
        return original_query

def write_expanded_queries(expanded_queries, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for query_id, expanded_query in expanded_queries.items():
            file.write(f"{query_id}\t{expanded_query}\n")

def main():
    queries = load_queries("nfcorpus/test.nontopic-titles.queries")
    knowledge_graph = load_knowledge_graph("gena_data_final_triples.csv")
    expanded_queries = {}
    for query_id, query in queries.items():
        entities = extract_entities(query, knowledge_graph)
        expanded_query = expand_query(entities, knowledge_graph, query)
        expanded_queries[query_id] = expanded_query
    write_expanded_queries(expanded_queries, "expanded_queries.txt")
    print("Expanded queries have been written to expanded_queries.txt")

if __name__ == "__main__":
    main()






