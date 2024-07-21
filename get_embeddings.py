from langchain_community.embeddings.ollama import OllamaEmbeddings

MODEL = "llama3"

# Get embeddings
def get_embeddings():
    embeddings = OllamaEmbeddings(model = MODEL)
    return embeddings