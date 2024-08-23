from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from whoosh.qparser import QueryParser
import os
import cProfile
from memory_profiler import profile

#@profile
def create_index(index_dir, nfdump_path, docdump_path, stopwords_path):
  # Define the schema for the index
  schema = Schema(id=ID(unique=True, stored=True),
                  url=TEXT(stored=True),
                  title=TEXT(stored=True),
                  abstract=TEXT(stored=True),
                  maintext=TEXT(stored=True),
                  comments=TEXT(stored=True),
                  topics_tags=TEXT(stored=True),
                  description=TEXT(stored=True),
                  doctors_note=TEXT(stored=True),
                  article_links=TEXT(stored=True),
                  question_links=TEXT(stored=True),
                  topic_links=TEXT(stored=True),
                  video_links=TEXT(stored=True),
                  medarticle_links=TEXT(stored=True))

  # Create the index directory if it doesn't exist
  if not os.path.exists(index_dir):
    os.mkdir(index_dir)

  # Create or open the index
  ix = index.create_in(index_dir, schema)

  # Load stop words from stopwords.large
  with open(stopwords_path, 'r', encoding='utf-8') as stopwords_file:
    stopwords = set(line.strip().lower() for line in stopwords_file)

  # Open corpus.txt for writing
  with open("corpus.txt", "w", encoding="utf-8") as corpus_file:
    # Get a writer
    writer = ix.writer()

    # Index nfdump.txt
    with open(nfdump_path, 'r', encoding='utf-8') as f:
      for line in f:
        fields = line.strip().split('\t')
        if len(fields) == 13:
          writer.add_document(id=fields[0], url=fields[1], title=fields[2], maintext=fields[3], comments=fields[4], topics_tags=fields[5], description=fields[6], doctors_note=fields[7],
          article_links=fields[8], question_links=fields[9], topic_links=fields[10], video_links=fields[11], medarticle_links=fields[12])

    # Index doc dump.txt
    with open(docdump_path, 'r', encoding='utf-8') as f:
      for line in f:
        fields = line.strip().split('\t')
        if len(fields) == 4:
          # Extract words from non-ID fields
          words = set(word for field in fields[1:] for word in field.lower().split())
          # Filter out stopwords
          content_words = words - stopwords
          # Write filtered words to corpus.txt
          for word in content_words:
            corpus_file.write(f"{word}\n")

          writer.add_document(id=fields[0],
                              url=fields[1],
                              title=fields[2],
                              abstract=fields[3])

    # Commit the changes and close the writer
    writer.commit()

  print("Indexing completed successfully. Content words stored in corpus.txt")

if __name__ == "__main__":
  # Define the directory paths
  index_dir = "indexing"
  nfdump_path = "nfcorpus/raw/nfdump.txt"
  docdump_path = "nfcorpus/raw/doc_dump.txt"
  stopwords_path = "nfcorpus/raw/stopwords.large"
  create_index(index_dir, nfdump_path, docdump_path, stopwords_path)
  '''with cProfile.Profile() as pr:
    create_index(index_dir, nfdump_path, docdump_path, stopwords_path)
  pr.print_stats()'''
