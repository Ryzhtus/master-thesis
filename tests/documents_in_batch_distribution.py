from named_entity_recognition.utils import create_dataset_and_document_dataloader
from transformers import BertTokenizer
import matplotlib.pyplot as plt

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

train_dataset, train_documents, train_dataloader = create_dataset_and_document_dataloader('conll',
                                                                                          '../data/conll2003/train.txt',
                                                                                          batch_size=32, shuffle=False,
                                                                                          tokenizer=bert_tokenizer)

eval_dataset, eval_documents, eval_dataloader = create_dataset_and_document_dataloader('conll',
                                                                                       '../data/conll2003/valid.txt',
                                                                                       batch_size=32, shuffle=False,
                                                                                       tokenizer=bert_tokenizer)

test_dataset, test_documents, test_dataloader = create_dataset_and_document_dataloader('conll',
                                                                                       '../data/conll2003/test.txt',
                                                                                       batch_size=32, shuffle=False,
                                                                                       tokenizer=bert_tokenizer)

train_documents_per_batch = []
eval_documents_per_batch = []
test_documents_per_batch = []

for batch in train_dataloader:
    document_ids = batch[3]
    train_documents_per_batch.append(len(set(document_ids)))

for batch in eval_dataloader:
    document_ids = batch[3]
    eval_documents_per_batch.append(len(set(document_ids)))

for batch in test_dataloader:
    document_ids = batch[3]
    test_documents_per_batch.append(len(set(document_ids)))



fig, axs = plt.subplots(1, 3, figsize=(20, 7))
axs[0].hist(train_documents_per_batch)
axs[0].set_title('Train')
axs[0].set_xlabel('Number of documents in batch')
axs[0].set_ylabel('Number of batches')
axs[1].hist(eval_documents_per_batch)
axs[1].set_title('Eval')
axs[1].set_xlabel('Number of documents in batch')
axs[1].set_ylabel('Number of batches')
axs[2].hist(test_documents_per_batch)
axs[2].set_title('Test')
axs[2].set_xlabel('Number of documents in batch')
axs[2].set_ylabel('Number of batches')
plt.savefig('document_per_batch_not_shuffle_standard_dataloader.png')
plt.show()