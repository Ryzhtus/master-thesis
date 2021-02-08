from named_entity_recognition.dataset import ConLL2003Dataset, read_data, create_dataset_and_dataloader
from named_entity_recognition.model import BertNER
from named_entity_recognition.train import train_model, test

from transformers import BertTokenizer
from transformers import AdamW
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
EPOCHS = 4
BATCH_SIZE = 16

sentences, tags, tags_number = read_data("data/conll2003/train.txt")
dataset = ConLL2003Dataset(sentences, tags, tags_number, TOKENIZER)

train_dataset, train_dataloader = create_dataset_and_dataloader("data/conll2003/train.txt", BATCH_SIZE, TOKENIZER)
eval_dataset, eval_dataloader = create_dataset_and_dataloader("data/conll2003/valid.txt", BATCH_SIZE, TOKENIZER)
test_dataset, test_dataloader = create_dataset_and_dataloader("data/conll2003/test.txt", BATCH_SIZE, TOKENIZER)

classes = len(dataset.ner_tags)

model = BertNER(classes).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)

train_model(model, criterion, optimizer, train_dataloader, eval_dataloader, train_dataset.tag2idx, DEVICE, EPOCHS)
test(model, criterion, test_dataloader, train_dataset.tag2idx, DEVICE)