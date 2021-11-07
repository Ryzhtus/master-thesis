from ner.dataset import ChunksPlusDocumentsDataset
from ner.reader import ReaderOntonotes
from ner.model import LongAttentionBERT

from torch.utils.data import DataLoader

from transformers import BertTokenizer, BigBirdConfig

BATCH_SIZE = 4
SHUFFLE = False
SEQ_LENGTH = 128 
PATH = "data/ontonotes/train"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

reader = ReaderOntonotes(include_document_ids=True)
sentences, labels, documents_masks, document2sentences, sentence2position = reader.read(PATH)
dataset = ChunksPlusDocumentsDataset(sentences, labels, SEQ_LENGTH, document2sentences, sentence2position, tokenizer, 'Bert')
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=SHUFFLE, collate_fn=dataset.paddings)

input_ids, label_ids, attention_mask, word_ids = next(iter(dataloader))

# количество классов для NER
classes = len(dataset.entity_tags)

config = BigBirdConfig()
config.pad_token_id = 0
config.bos_token_id = 101
config.sep_token_id = 102
config.eos_token_id = -1
config.attention_type='original_full'

model = LongAttentionBERT(model_name='bert-base-cased', classes=classes, attention_config=config)

print(model.forward(input_ids=input_ids, attention_mask=attention_mask))