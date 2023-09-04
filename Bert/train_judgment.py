# -*- coding:utf-8 -*-
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 准备数据
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 定义训练参数
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_encodings = tokenizer("Hello, my dog is cute", return_tensors="pt", padding=True, truncation=True)
train_labels = torch.tensor([1]).unsqueeze(0)
train_dataset = MyDataset(train_encodings, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_dataloader) * 3 // 2  # 训练轮数
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
for epoch in range(3):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# 测试模型
model.eval()
test_encodings = tokenizer("Hello, my cat is cute", return_tensors="pt", padding=True, truncation=True)
test_labels = torch.tensor([0]).unsqueeze(0)
test_dataset = MyDataset(test_encodings, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

predictions, true_labels = [], []
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(true_labels.cpu().numpy())

print("Accuracy:", accuracy_score(true_labels, predictions))
print("Precision:", precision_recall_fscore_support(true_labels, predictions, average='weighted'))
