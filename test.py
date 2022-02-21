from ner_lightning.dataset import ChunksDataset
from ner_lightning.reader import ReaderCoNLL

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

reader = ReaderCoNLL()
documents = reader.parse("data/conll2003/IOB2/train.txt")
dataset = ChunksDataset(documents, 256, tokenizer)
print(len(dataset.entity_tags), dataset.entity_tags)