from fastembed import TextEmbedding
import chromadb
import anthropic

embedder: TextEmbedding = None
chroma_client: chromadb.HttpClient = None
collection = None
anthropic_client: anthropic.Anthropic = None
