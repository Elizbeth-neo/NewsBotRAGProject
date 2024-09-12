import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
import config
from langchain.vectorstores import Pinecone as PL
from pinecone import Pinecone, ServerlessSpec
from loguru import logger as logging

config = config.Config()
TOP_K = config.retriever_top_k
DATA_PATH = os.getenv('DATA_PATH')

def load_and_prepare_documents():
    news_df = pd.read_csv(os.getenv(DATA_PATH))
    loader = DataFrameLoader(news_df, page_content_column='text')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=config.chunk_size,
                                          chunk_overlap=config.chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

class ChromaDb:
    def __init__(self, index_name, embedding_function):
        self.embeddings = embedding_function
        self.index_name = index_name
        self.db = self.load_index()

    def load_index(self):
        pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        )
        if self.index_name not in pc.list_indexes().names():
            logging.info(f'Creating new indexes')
            docs = load_and_prepare_documents()
            pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=os.getenv('PINECONE_API_CLOUD'),
                    region=os.getenv('PINECONE_API_ENV')
                )
            )
            docsearch = PL.from_documents(docs, self.embeddings, self.index_name)
        else:
            logging.info(f'Importing indexes from Pinecone index "{self.index_name}"')
            docsearch = PL.from_existing_index(self.index_name, self.embeddings)

        return docsearch

class FaissDb:
    def __init__(self, index_path, model_name=config.embedding_rubert):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.index_path = index_path
        self.db = self.load_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            return FAISS.load_local(self.index_path,
                                    self.embeddings,
                                    allow_dangerous_deserialization=True
                                    )
        else:
            self.docs = load_and_prepare_documents()
            self.docsearch = FAISS.from_documents(self.docs,
                                                  self.embeddings)
            self.docsearch.save_local(self.index_path)
            return self.docsearch


    def similarity_search(self, question: str, k: int = TOP_K):
        retrieved_docs = self.db.similarity_search(question, k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
