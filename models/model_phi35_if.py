from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import is_flash_attn_2_available

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model_phi3_5_if:
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
                "You are a professional news editor. Your task is to: \n"
                "1. **Remove all AI artifacts**: Delete word counts (e.g., '(Word Count: ≈250)'), language switches, slashes, or unrelated notes. \n"
                "2. **Maintain factual integrity**: Keep all names, dates, and key details unchanged. \n"
                "3. **Rewrite logically**: Smooth transitions, fix awkward phrasing, and ensure coherence. \n"
                "4. **Enhance the story**: \n"
                "   - Add expert quotes (attribute to real/fictional experts like 'Dr. X, economist at Y University'). \n"
                "   - Include background context (historical trends, related events). \n"
                "   - Expand with new facts (statistics, official statements). \n"
                "5. **Style**: Use formal journalistic tone (AP Style). Avoid first-person ('I'). \n\n"
                "You can write an article expanding its meaning!\n\n"
                "Return ONLY the polished article, no meta-commentary.\n\n"
                "**Return ONLY PAPER TEXT!**\n\n"
                "**Do not start whoth word 'in'!**"
            )
        },
        {
            "role": "user", 
            "content":news_paper
        }
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tokenize_special_tokens=True,
        )    
    
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
        min_new_tokens = min(1024, int(0.6 * cnt))
        max_new_tokens = min(1600, int(1.3 * cnt))
        
        # Обновляем параметры генерации
        generation_args = {
            **self.generation_args,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
        }
        
        output = self.pipe(prompt, **generation_args)
        return output[0]["generated_text"]