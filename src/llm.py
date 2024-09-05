import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)

class LLMChatModel:
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cuda"):
        HF_ACCESS_TOKEN=os.getenv('HUGGINGFACE_API_TOKEN')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=CACHE_DIR, token=HF_ACCESS_TOKEN
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=HF_ACCESS_TOKEN,
        )
        self.model.eval()
        self.chat = []
        self.device = device

    def generate(self, question: str, context: str = None, max_new_tokens: int = 250):
        if context == None or context == "":
            prompt = f"""Tell me you didn't get the context and can't answer the question: {question}"""
        else:
            prompt = f"""You are a Russian-speaking news correspondent. People will ask you questions about the news.
        Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer from the context but you may want to check the following links". Give the answer on Russian..
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
                do_sample=False,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt):]
        response = response.replace("<eos>", "")

        return response