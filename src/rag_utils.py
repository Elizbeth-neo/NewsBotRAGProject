import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader



CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

class FaissDb:
    def __init__(self, index_path, model_name="cointegrated/rubert-tiny2"):
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
            self.load_and_prepare_documents()
            self.docsearch = FAISS.from_documents(self.docs,
                                                  self.embeddings
                                                 )
            self.docsearch.save_local(self.index_path)
            return self.docsearch

    def load_and_prepare_documents(self):
        news_df = pd.read_csv(os.getenv('DATA_PATH'))
        loader = DataFrameLoader(news_df, page_content_column='text')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        self.docs = text_splitter.split_documents(documents)


    def similarity_search(self, question: str, k: int = 5):
        retrieved_docs = self.db.similarity_search(question, k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
