from ner.utils import create_dataset_and_document_dataloader

from transformers import BertTokenizer

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
BATCH_SIZE = 32

train_dataset, train_documents, train_dataloader = create_dataset_and_document_dataloader('conll',
                                                                                          'data/conll2003/train.txt',
                                                                                          batch_size=BATCH_SIZE,
                                                                                          shuffle=False,
                                                                                          tokenizer=TOKENIZER)

eval_dataset, eval_documents, eval_dataloader = create_dataset_and_document_dataloader('conll',
                                                                                       'data/conll2003/valid.txt',
                                                                                       batch_size=BATCH_SIZE,
                                                                                       shuffle=False,
                                                                                       tokenizer=TOKENIZER)

test_dataset, test_documents, test_dataloader = create_dataset_and_document_dataloader('conll',
                                                                                       'data/conll2003/test.txt',
                                                                                       batch_size=BATCH_SIZE,
                                                                                       shuffle=False,
                                                                                       tokenizer=TOKENIZER)

print(len(train_dataloader))
print(len(eval_dataloader))
print(len(test_dataloader))