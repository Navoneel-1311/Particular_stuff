from whoosh import index
from whoosh.qparser import QueryParser
import os

def nfdump_search(index_dir, query_id):
    # Open the existing index
    ix = index.open_dir(index_dir)

    # Create a searcher object
    searcher = ix.searcher()

    # Define the query parser
    parser = QueryParser("id", ix.schema)

    # Parse the user's query
    query = parser.parse(query_id)

    # Perform the search
    results = searcher.search(query, limit=None)

    # Display the results
    print("Number of results:", len(results))
    for hit in results:
        print("ID:", hit["id"])
        print("URL:", hit["url"])
        print("Title:", hit["title"])
        print("Main Text:", hit["maintext"])
        print("Comments:", hit["comments"])
        print("Topics Tags:", hit["topics_tags"])
        print("Description:", hit["description"])
        print("Doctors Note:", hit["doctors_note"])
        print("Article Links:", hit["article_links"])
        print("Question Links:", hit["question_links"])
        print("Topic Links:", hit["topic_links"])
        print("Video Links:", hit["video_links"])
        print("Medical Article Links:", hit["medarticle_links"])
        print()

def doc_dump_search(index_dir, query_id):
    # Open the existing index
    ix = index.open_dir(index_dir)

    # Create a searcher object
    searcher = ix.searcher()

    # Define the query parser
    parser = QueryParser("id", ix.schema)

    # Parse the user's query
    query = parser.parse(query_id)

    # Perform the search
    results = searcher.search(query, limit=None)

    # Display the results
    print("Number of results:", len(results))
    for hit in results:
        print("ID:", hit["id"])
        print("URL:", hit["url"])
        print("Title:", hit["title"])
        print("Abstract:", hit["abstract"])
        print()

if __name__ == "__main__":
    # Define the directory path where the index is located
    index_dir = "indexing"

    # Define the query ID for nfdump
    nf_query_id = input("Enter the ID to search in nfdump: ")
    
    # Define the query ID for doc_dump
    doc_query_id = input("Enter the ID to search in doc_dump: ")

    # Perform the search and display the results
    nfdump_search(index_dir, nf_query_id)
    doc_dump_search(index_dir, doc_query_id)
