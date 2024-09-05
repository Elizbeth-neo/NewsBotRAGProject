import os
from llm import LLMChatModel
import rag_utils

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

class ChatBot:
    def __init__(self):
        self.model = self.load_model()
        self.db = self.load_db()
        self.messages = []

    def load_model(self):
        model = LLMChatModel(model_id="google/gemma-2b-it", device="cuda")
        return model

    def load_db(self):
        index_path = os.getenv('FAISS_INDEX_PATH')
        return rag_utils.FaissDb(index_path=index_path, model_name="cointegrated/rubert-tiny2")

    def ask_question(self, question, max_new_tokens=512, k=3):
        self.messages.append({"role": "user", "content": question})
        user_prompt = self.messages[-1]["content"]
        context = (
            self.db.similarity_search(user_prompt, k=k)
        )
        answer = self.model.generate(question=user_prompt, context=context, max_new_tokens=max_new_tokens)
        self.messages.append({"role": "assistant", "content": answer})
        return answer

def main():
    chatbot = ChatBot()
    print("ChatBot is ready. Type 'exit' to quit.")

    while True:
        user_input = input("Ask a question (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        else:
            answer = chatbot.ask_question(user_input)
            print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
