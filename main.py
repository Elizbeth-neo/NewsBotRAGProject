import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as P
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS

class ChatBot:
    def __init__(self):
        load_dotenv()

        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        self.index_name = os.environ.get("PINECONE_API_INDEX_NAME")
        self.region = os.getenv('PINECONE_API_ENV')
        self.cloud = os.getenv('PINECONE_API_CLOUD')
        self.index_path = os.getenv('FAISS_INDEX_PATH')
        self.storage_type = os.getenv('STORAGE_TYPE')

        self.embeddings = HuggingFaceEmbeddings()
        self.ru_embeddings = HuggingFaceEmbeddings(model_name='cointegrated/rubert-tiny2')

        if self.storage_type == 'faiss':
            if os.path.exists(self.index_path):
                self.docsearch = FAISS.load_local(self.index_path, self.ru_embeddings, allow_dangerous_deserialization=True)
            else:
                self.load_and_prepare_documents()
                self.docsearch = FAISS.from_documents(self.docs, self.ru_embeddings)
                self.docsearch.save_local(self.index_path)

        elif self.storage_type == 'pinecone':
            if self.index_name not in self.pc.list_indexes().names():
                self.load_and_prepare_documents()
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.region
                    )
                )
                self.docsearch = P.from_documents(self.docs, self.embeddings, index_name=self.index_name)
            else:
                self.docsearch = P.from_existing_index(self.index_name, self.embeddings)

        self.retriever = self.docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.8,
            top_k=50,
            max_new_tokens=1024,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
            model_kwargs={"clean_up_tokenization_spaces":True}
        )

        template = """
        You are a Russian-speaking news correspondent. People will ask you questions about the news.
        Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
        2. If you find the answer, write the answer in a concise way with five sentences maximum.
        3. Give the answer on Russian.
        
        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def load_and_prepare_documents(self):
        news_df = pd.read_csv(os.getenv('DATA_PATH'))
        loader = DataFrameLoader(news_df, page_content_column='text')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        self.docs = text_splitter.split_documents(documents)

    def ask_question(self, question):
        result = self.qa.invoke({"query": question})
        if len(result['result']) < 5 or result==' ':
            return 'API модели не отработало'
        return result['result']

def main():
    chatbot = ChatBot()

    while True:
        user_input = input("Ask a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            answer = chatbot.ask_question(user_input)
            print(f"Answer: {answer}")


if __name__ == "__main__":
    main()


