from named_entity_recognition.dataset import DatasetNER, read_conll, create_dataset_and_dataloader
from named_entity_recognition.dataset import convert_to_document, make_sentences_mask
from named_entity_recognition.model import BertNER
from named_entity_recognition.train import train_model, eval_epoch

from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
EPOCHS = 4
BATCH_SIZE = 16

sentences, tags = read_conll("../data/conll2003/train.txt")
documents = convert_to_document(sentences, tags)
sentences, tags, masks = make_sentences_mask(documents)
dataset = DatasetNER(sentences, tags, masks, TOKENIZER)

train_dataset, train_dataloader = create_dataset_and_dataloader("../data/conll2003/train.txt", BATCH_SIZE, TOKENIZER)
eval_dataset, eval_dataloader = create_dataset_and_dataloader("../data/conll2003/valid.txt", BATCH_SIZE, TOKENIZER)
test_dataset, test_dataloader = create_dataset_and_dataloader("../data/conll2003/test.txt", BATCH_SIZE, TOKENIZER)

classes = len(dataset.ner_tags)

model = BertNER(classes).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(DEVICE)

train_model(model, criterion, optimizer, train_dataloader, eval_dataloader, train_dataset.tag2idx, train_dataset.idx2tag, DEVICE, None, EPOCHS)
eval_epoch(model, criterion, test_dataloader, train_dataset.tag2idx, train_dataset.idx2tag, DEVICE)