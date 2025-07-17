import os
import glob
from sentence_transformers import SentenceTransformer, util
import torch



# Step 1: Load and chunk the Markdown files
def load_and_chunk_markdown(folder_path, chunk_size=300):
    all_chunks = []
    chunk_sources = []

    for filepath in glob.glob(os.path.join(folder_path, '*.md')):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Simple chunking by paragraphs (can be improved later)
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50:  # ignore very short lines
                all_chunks.append(para.strip())
                chunk_sources.append(filepath)

    return all_chunks, chunk_sources

# Step 2: Embed chunks with SentenceTransformer
def embed_chunks(model, chunks):
    return model.encode(chunks, convert_to_tensor=True)

# Step 3: Accept a user query and embed it
def embed_query(model, query):
    return model.encode(query, convert_to_tensor=True)

# Step 4: Retrieve top-k relevant chunks
def retrieve_chunks(query_embedding, chunk_embeddings, chunks, sources, top_k=3):
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)[0]
    results = []

    for hit in hits:
        idx = hit['corpus_id']
        score = hit['score']
        results.append({
            'chunk': chunks[idx],
            'source': sources[idx],
            'score': score
        })

    return results

# Step 5: Answer generation using chunks (simple template for now)
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join([f"[{i+1}] {item['chunk']}" for i, item in enumerate(retrieved_chunks)])
    print("\n--- Context (Top-k Chunks) ---")
    print(context)
    print("\n--- Answer ---")
    print(f"Based on the above information, here's an answer to your question:\n{query}")
    print("\nSources:")
    for i, item in enumerate(retrieved_chunks):
        print(f"[{i+1}] {item['source']} (score: {item['score']:.2f})")

# ---------- MAIN ----------
if __name__ == "__main__":
    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Load and chunk the data
    print("Loading and chunking Markdown files...")
    chunks, sources = load_and_chunk_markdown("data")

    print(f"Loaded {len(chunks)} chunks.")

    # Embed all chunks
    print("Embedding chunks...")
    chunk_embeddings = embed_chunks(model, chunks)

    while True:
        query = input("\n🔍 Enter your question (or 'exit'): ").strip()
        if query.lower() == 'exit':
            break

        # Embed query
        query_embedding = embed_query(model, query)

        # Retrieve top-k chunks
        top_chunks = retrieve_chunks(query_embedding, chunk_embeddings, chunks, sources, top_k=3)

        # Generate and print answer
        generate_answer(query, top_chunks)
