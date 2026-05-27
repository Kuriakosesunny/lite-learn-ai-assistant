from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "uploads")

INDEX_PATH = os.path.join(UPLOAD_DIR, "faiss.index")
IDMAP_PATH = os.path.join(UPLOAD_DIR, "faiss_ids.pkl")

# make sure uploads folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# global variables
index = None
chunk_ids = []


# -----------------------------------
# load or create FAISS index
# -----------------------------------
def load_or_create_index():

    global index
    global chunk_ids

    if index is not None:
        return index

    # Get the dimension directly from the model to avoid mismatches
    dimension = model.get_sentence_embedding_dimension()

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        if os.path.exists(IDMAP_PATH):
            with open(IDMAP_PATH, "rb") as f:
                chunk_ids = pickle.load(f)

    else:
        index = faiss.IndexFlatL2(dimension)
        chunk_ids = []

    return index


# -----------------------------------
# add embeddings to FAISS
# -----------------------------------
def add_embeddings(texts, ids):

    global chunk_ids

    index = load_or_create_index()

    # create embeddings
    embeddings = model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    # add vectors
    index.add(embeddings)

    # store corresponding DB chunk ids
    chunk_ids.extend(ids)

    # save index
    faiss.write_index(index, INDEX_PATH)

    # save id mapping
    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(chunk_ids, f)


# -----------------------------------
# search FAISS
# -----------------------------------
def search(question, k=5):

    global chunk_ids

    if not os.path.exists(INDEX_PATH):
        return []

    index = load_or_create_index()

    # embed question
    q_embedding = model.encode([question])
    q_embedding = np.array(q_embedding).astype("float32")

    # search
    distances, indices = index.search(q_embedding, k)

    results = []

    for i in indices[0]:
        if i < len(chunk_ids):
            results.append(chunk_ids[i])

    return results