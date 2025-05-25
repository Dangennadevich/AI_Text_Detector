from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import is_flash_attn_2_available

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_phi3_5:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct", device=device):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=2048)
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "no_repeat_ngram_size":2, 
        }

    def build_prompt(self, news_paper: str) -> str:
        """Создаем промпт для phi 3.
        
        news_paper: Статья, которую будем переписывать от лица модели

        output: Статья написанная моделью
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "You - AI-asistant - Author of news!"
                    "Rewrite the following news article in your own words, based on the facts, names, and details from the text. Expand the story by adding relevant new information - come up with expert quotes, add the journalist's reasoning, and new facts. Invent a backstory for the news, new characters, and more."
                )
            },
            {"role": "user", "content": news_paper}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tokenize_special_tokens=True,
        )      

    def build_prompts(self, texts: list[str]) -> list[str]:
        """Создаем промпты для батча
        
        texts: Список статей, котоыре будем переписывать от лица модели
        """
        prompts = []
        for text in texts:
            messages = [
                {
                    "role": "system",
                    "content": "Rewrite the following news article in your own words, based on the facts, names, and details from the text. Expand the story by adding relevant new information - come up with expert quotes, add a journalist's reasoning, and new facts. Write a big text!"
                },
                {"role": "user", "content": text}
            ]
            prompts.append(self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        return prompts
    
    @torch.no_grad()
    def model_inference(self, news_paper:str) -> str:
        """Инференс PHI 3.5

        texts: список статей для обработки

        out: Список статей от лица модели
        """
        prompt = self.build_prompt(news_paper=news_paper)
        
        # Токенизируем промпт для подсчета длины
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(device)
        cnt = encoded_input.input_ids.shape[1]
        
        # Рассчитываем динамические лимиты кол-ва токенов 
        min_new_tokens = min(1024, int(0.7 * cnt))
        max_new_tokens = min(1024, int(0.9 * cnt))

        print('cnt, min_new_tokens, max_new_tokens')
        print(cnt, min_new_tokens, max_new_tokens)
        
        # Обновляем параметры генерации
        generation_args = {
            **self.generation_args,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
        }
        
        output = self.pipe(prompt, **generation_args)
        return output[0]["generated_text"]

    @torch.no_grad()
    def batch_inference(self, texts: list[str], batch_size=8) -> list[str]:
        """Групповая бработка запросов
        
        texts: список статей для обработки
        batch_size: размер батча
        
        out: Список статей от лица модели
        """
        all_outputs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            prompts = self.build_prompts(batch)
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2046,
                return_token_type_ids=False
            ).to(self.model.device)
            
            # Получаем длину каждого элемента в батче
            input_lengths = [len(seq) for seq in inputs.input_ids]
            
            # Рассчитываем параметры генерации для каждог элемента батча
            min_new_tokens = min(1024, int(0.7 * input_lengths))
            max_new_tokens = min(1024, int(0.9 * input_lengths))
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            decoded = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            all_outputs.extend(decoded)
        
        return all_outputs