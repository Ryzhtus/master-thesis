from ner.utils import create_chunk_dataset_and_document_dataloader

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
BATCH_SIZE = 32

train_dataset, train_documents, train_dataloader = create_chunk_dataset_and_document_dataloader('conll', 'data/conll2003/IOB/eng.train',
                                                                                                128, 32, False, tokenizer)

eval_dataset, eval_documents, eval_dataloader = create_chunk_dataset_and_document_dataloader('conll', 'data/conll2003/IOB/eng.testa',
                                                                                                128, 32, False, tokenizer)

test_dataset, test_documents, test_dataloader = create_chunk_dataset_and_document_dataloader('conll', 'data/conll2003/IOB/eng.testb',
                                                                                                128, 32, False, tokenizer)

print(train_dataset.entity_tags)
print(eval_dataset.entity_tags)
print(test_dataset.entity_tags)

test_dataset.entity_tags = train_dataset.entity_tags
eval_dataset.entity_tags = train_dataset.entity_tags

test_dataset.tag2idx = train_dataset.tag2idx
eval_dataset.tag2idx = test_dataset.tag2idx

test_dataset.idx2tag = train_dataset.idx2tag
eval_dataset.idx2tag = train_dataset.idx2tag

for i in train_dataset.idx2tag.keys():
    print(i, train_dataset.idx2tag[i], eval_dataset.idx2tag[i])


#assert train_dataset.entity_tags == eval_dataset.entity_tags
#assert train_dataset.entity_tags == test_dataset.entity_tags
#assert eval_dataset.entity_tags == test_dataset.entity_tags

assert train_dataset.idx2tag == eval_dataset.idx2tag
assert train_dataset.idx2tag == test_dataset.idx2tag
assert eval_dataset.idx2tag == test_dataset.idx2tag

assert train_dataset.tag2idx == eval_dataset.tag2idx
assert train_dataset.tag2idx == test_dataset.tag2idx
assert eval_dataset.tag2idx == test_dataset.tag2idx