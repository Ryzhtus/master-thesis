from ner.utils import create_dataset_and_document_dataloader
from ner.reader import ReaderCoNLL

from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
BATCH_SIZE = 32

reader = ReaderCoNLL(include_document_ids=True)
sentences, labels, _, document2sentences, sentence2position = reader.read('data/conll2003/valid.txt')

print(len(document2sentences))
