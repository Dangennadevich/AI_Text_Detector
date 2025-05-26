from transformers import BertTokenizerFast, BertModel
from torch import nn

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(device)

MODEL_PATH = "/app/model.pt"

assert len(MODEL_PATH)>0, 'PATH модели не заполнен'

# Соберем модель
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
bert = BertModel.from_pretrained(MODEL_NAME)

# Инициализация модели
class BertBinaryClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size=768, dropout=0.2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)  # 2 класса

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # берём [CLS] токен: outputs.last_hidden_state[:,0,:]
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}
    
# Загружаем весов в модель, перенос на device и режим eval
checkpoint = torch.load(
    MODEL_PATH,
    weights_only=False,
    # map_location=torch.device(device)
)

model = BertBinaryClassifier(bert)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()