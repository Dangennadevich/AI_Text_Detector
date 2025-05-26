from model import model, tokenizer
from typing import Optional

import torch
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_predictions.log')
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def predict_paper(paper: str, tokenizer=tokenizer, model=model) -> Optional[float]:
    """
    Определение вероятности генерации статьи при помощи LLM
    
    Args:
        paper: Текст новостной статьи
        tokenizer: bert-base-uncased tokenizer
        model: BertBinaryClassifier
    
    Returns:
        float: Вероятность принадлежности к классу 1
        None: В случае ошибки
    
    Raises:
        ValueError: Если входной текст пустой
    """
    try:
        # Проверка входных данных
        if not paper or not isinstance(paper, str):
            logger.error("Input text is empty or not a string")
            raise ValueError("Input text must be a non-empty string")
        
        if len(paper) < 10:
            logger.warning(f"Very short input text (length: {len(paper)})")

        logger.info("Starting prediction process...")
        
        # Токенизация
        logger.debug("Tokenizing input text...")
        encoding = tokenizer(
            paper,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Перенос на устройство
        logger.debug("Moving tensors to device...")
        item = {key: val.to(device) for key, val in encoding.items()}
        
        # Предсказание
        logger.info("Running model inference...")
        with torch.no_grad():
            try:
                out = model(**item)
                logits = out['logits']
                
                # Вычисление вероятностей
                probs = torch.softmax(logits, dim=1)
                prob_1_class = probs[:, 1].detach().cpu().numpy()
                
                logger.info(f"Prediction completed successfully. Probability: {prob_1_class[0]:.4f}")
                return float(prob_1_class[0])
                
            except RuntimeError as e:
                logger.error(f"Model inference failed: {str(e)}")
                if "CUDA out of memory" in str(e):
                    logger.error("CUDA out of memory. Try reducing batch size or model size.")
                return None
                
    except Exception as e:
        logger.exception(f"Unexpected error during prediction: {str(e)}")
        return None
