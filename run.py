from ner.utils import create_dataset_and_document_dataloader
from ner.reader import ReaderCoNLL

from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
BATCH_SIZE = 32

train_dataset, train_documents, train_dataloader = create_dataset_and_document_dataloader('conll', 'data/conll2003/train.txt', 32, True, TOKENIZER)

print(train_dataset.idx2tag)
