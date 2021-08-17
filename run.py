from ner.utils import create_chunk_dataset_and_document_dataloader

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
BATCH_SIZE = 32

train_dataset, train_documents, train_dataloader = create_chunk_dataset_and_document_dataloader('conll', 'data/conll2003/train.txt',
                                                                                                128, 32, False, tokenizer)

eval_dataset, eval_documents, eval_dataloader = create_chunk_dataset_and_document_dataloader('conll', 'data/conll2003/valid.txt',
                                                                                                128, 32, False, tokenizer)

test_dataset, test_documents, test_dataloader = create_chunk_dataset_and_document_dataloader('conll', 'data/conll2003/test.txt',
                                                                                                128, 32, False, tokenizer)