from fastembed import TextEmbedding
from qdrant_client import QdrantClient
import anthropic

embedder: TextEmbedding = None
qdrant_client: QdrantClient = None
anthropic_client: anthropic.Anthropic = None
