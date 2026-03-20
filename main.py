from sentence_transformers import SentenceTransformer, util
import numpy as np
from data import sentences

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert sentences to embeddings
sentence_embeddings = model.encode(sentences)

# User query
query = input("Enter your search query: ")

# Convert query to embedding
query_embedding = model.encode(query)

# Compute similarity
similarities = util.cos_sim(query_embedding, sentence_embeddings)

# Convert to numpy
similarities = similarities.cpu().numpy()

# Get top 3 results
top_results = np.argsort(similarities[0])[-3:][::-1]

print("\nTop Results:")
for idx in top_results:
    print(sentences[idx])