from enum import Enum


class ModelProvider(str, Enum):
    # LLM
    DEEPSEEK = "deepseek"

    # Embedding
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    VOYAGE = "voyage"

    # OCR
    DATALAB = "datalab"
    MISTRAL = "mistral"
    LLAMAPARSE = "llamaparse"

    # Audio
    DEEPGRAM = "deepgram"
    GLADIA = "gladia"
    ASSEMBLYAI = "assemblyai"
    GROQ = "groq"
    ELEVENLABS = "elevenlabs"

    # Video
    TWELVELABS = "twelvelabs"

    # Reranker
    # XENOVA = "xenova"
    # BBAI = "baai"
    JINA = "jina"
    COHERE = "cohere"

    # Sparse
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    PRITHIVIDA = "prithivida"
    PRITHVIDA = "prithvida"
