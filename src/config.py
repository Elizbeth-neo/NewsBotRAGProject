import os

class Config():
    def  __init__(self):
        self.cache_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        )
        self.history_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat-history.txt")
        )

        #--- Indexes ---
        self.embedding_bge = "BAAI/bge-m3"
        self.embedding_rubert = "cointegrated/rubert-tiny2"
        self.retriever_top_k = 3
        self.chunk_size = 500
        self.chunk_overlap = 280

        #--- LLMs ---
        # from ollama
        self.device = "cuda"
        self.llama_model = "llama3.1"       #"qwen2:7b"
        self.ollama_url = "http://localhost:11434"

        # from transformers
        self.gemma_model = "google/gemma-2b-it"
        self.gemma_max_tokens = 250
        self.gemma_top_k = 50
        self.gemma_top_p = 0.85
        self.gemma_temperature = 0.6



