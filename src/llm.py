import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import config

load_dotenv()

config = config.Config()
TOP_K = config.retriever_top_k
HF_ACCESS_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

class BGE:
    def __init__(self, model_id: str = config.embedding_bge, device: str = config.device):
        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name=model_id,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

class OllamaLLM:
    def __init__(self, model_id: str = config.llama_model,
                 device: str = config.device,
                 url: str = config.ollama_url):
        self.model_id = model_id
        self.model_url = url
        self.chat = []
        self.device = device
        self.model = Ollama(
            model=self.model_id,
            base_url=self.model_url,
            verbose=True
        )


    def generate(self, question, indexdb, k=TOP_K):
        template = """
        You are a Russian-speaking news correspondent. People will ask you questions about the news.
        Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
        2. If you find the answer, write the answer in a concise way with five sentences maximum.
        3. Give the answer on Russian.
        4. Keep a formal communication style
    
        Context: {context}
        Question: {question}
        Answer:
        """

        formatted_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        conversation = ConversationalRetrievalChain.from_llm(
            self.model,
            retriever=indexdb.db.as_retriever(search_type="similarity",
                                           search_kwargs={"k": k}),
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": formatted_prompt}
        )

        response = conversation.invoke({"question": question, "chat_history": self.chat})

        return response

class LLMChatModel:
    def __init__(self, model_id: str = config.gemma_model, device=config.device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=config.cache_dir, token=HF_ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=config.cache_dir,
            token=HF_ACCESS_TOKEN,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = config.gemma_max_tokens):
        if context == None or context == "":
            prompt = f"""Tell me you didn't get the context and can't answer the question: {question}"""
        else:
            prompt = f"""You are a Russian-speaking news correspondent. People will ask you questions about the news.
        Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. 
        Just say "I can't find the final answer from the context but you may want to check the following links". 
        Give the answer on Russian.
        2. If you find the answer, write the answer in a concise way with five sentences maximum.
        3. Give the answer on Russian.
        Context: {context}.
        Question: {question}
        """
        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(formatted_prompt)

        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=config.gemma_top_k,
                top_p=config.gemma_top_p,
                temperature=config.gemma_top_p
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt):]
        response = response.replace("<eos>", "")

        return response

