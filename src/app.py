import os
import llm
import rag_utils
from loguru import logger as logging
from config import Config

INDEX_NAME = os.getenv('INDEX_NAME')
TOP_K = Config().retriever_top_k
HISTORY_DIR = Config().history_dir
MODEL_ID = Config().llama_model
DEVICE = Config().device

class ChatBot:
    def __init__(self):
        self.model = self.load_model()
        self.embedding_function = self.load_embedding_model().embedding_function
        self.db = self.load_db()
        self.messages = []

    def load_model(self):
        model = llm.OllamaLLM(model_id=MODEL_ID, device=DEVICE)
        logging.info(f'run with "{MODEL_ID}"')
        return model

    def load_embedding_model(self):
        embedding_model = llm.BGE()
        return embedding_model

    def load_db(self):
        logging.info(f'Started load ChromaDb')
        chroma_instance = rag_utils.ChromaDb(index_name=INDEX_NAME, embedding_function=self.embedding_function)
        return chroma_instance

    def ask_question(self, question):
        response = self.model.generate(question=question, indexdb=self.db, k=TOP_K)
        self.messages.append({"role": "user",
                              "question": response['question'],
                              "content": [r.metadata for r in response['source_documents']]})
        self.messages.append({"role": "assistant", "content": response['answer']})
        return response['answer']

def main():
    chatbot = ChatBot()
    logging.info("ChatBot is ready. Type 'exit' to quit.")

    while True:
        user_input = input("Ask a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            history = chatbot.messages
            with open(HISTORY_DIR, "w") as text_file:
                text_file.write(str(history))
            logging.info(f'Chat history saved at: {HISTORY_DIR}')
            break
        else:
            answer = chatbot.ask_question(user_input)
            print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
