from model import model, tokenizer

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(device)

def predict_paper(paper, tokenizer=tokenizer, model=model):
  '''Функция для определение вероятности генерации статьи при помощи LLM
  
    paper: Текст новостной статьи
    tokenizer: bert-base-uncased tokenizer
    model: BertBinaryClassifier
  
    Return : float 
  '''
  encoding = tokenizer(
    paper,
    add_special_tokens=True,
    truncation=True,
    max_length=512,
    padding='max_length',
    return_tensors='pt'
  )

  item = {key: val.to(device) for key, val in encoding.items()}

  with torch.no_grad():
    out = model(**item)

  logits = out['logits']
  prob_1_class = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

  return float(prob_1_class[0])
