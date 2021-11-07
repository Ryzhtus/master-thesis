from ner.dataset import ChunksPlusDocumentsDataset
from ner.reader import ReaderOntonotes

from torch.utils.data import DataLoader

from transformers import BertTokenizer

BATCH_SIZE = 4
SHUFFLE = False
SEQ_LENGTH = 128 
PATH = "data/ontonotes/train"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

reader = ReaderOntonotes(include_document_ids=True)
sentences, labels, documents_masks, document2sentences, sentence2position = reader.read(PATH)
dataset = ChunksPlusDocumentsDataset(sentences, labels, SEQ_LENGTH, document2sentences, sentence2position, tokenizer, 'Bert')
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=SHUFFLE, collate_fn=dataset.paddings)

print(dataset[0])