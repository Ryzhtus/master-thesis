import matplotlib.pyplot as plt

from named_entity_recognition.iterator import DocumentBatchIterator
from named_entity_recognition.dataset import DatasetDocumentNER
from named_entity_recognition.reader import ReaderDocumentCoNLL

from transformers import BertTokenizer

paths = ['../data/conll2003/train.txt', '../data/conll2003/valid.txt', '../data/conll2003/test.txt']
subsets = ['Train', 'Valid', 'Test']
data = {}
documents_lengths = {}
batch_length_1_number = {subset: 0 for subset in subsets}

for subset, filepath in zip(subsets, paths):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    reader = ReaderDocumentCoNLL()
    sentences, tags, masks, document2sentences = reader.get_sentences(filepath)
    dataset = DatasetDocumentNER(sentences, tags, masks, tokenizer)
    data_iterator = DocumentBatchIterator(dataset, document2sentences, group_documents=True,
                                          batch_size=32, shuffle=False)

    length_per_document = [len(document2sentences[key]) for key in document2sentences.keys()]
    for key in document2sentences.keys():
        document = document2sentences[key]
        if (len(document) // 32) > 0:
            batch_length_1_number[subset] += len(document) // 32

    sentences2documents = {}
    for key in document2sentences.keys():
        sentences_ids = document2sentences[key]
        for id in sentences_ids:
            sentences2documents[id] = key

    batches = data_iterator.group_ids()
    batches_document_count = []

    for batch in batches:
        document_ids = []

        for idx in range(len(batch)):
            document_ids.append(sentences2documents[batch[idx]])

        document_ids = set(document_ids)

        batches_document_count.append(len(document_ids))

    data[subset] = batches_document_count
    documents_lengths[subset] = length_per_document



fig, axs = plt.subplots(1, 3, figsize=(20, 7))
axs[0].hist(data[subsets[0]])
axs[0].set_title(subsets[0])
axs[0].set_xlabel('Number of documents in batch')
axs[0].set_ylabel('Number of batches')
axs[1].hist(data[subsets[1]])
axs[1].set_title(subsets[1])
axs[1].set_xlabel('Number of documents in batch')
axs[1].set_ylabel('Number of batches')
axs[2].hist(data[subsets[2]])
axs[2].set_title(subsets[2])
axs[2].set_xlabel('Number of documents in batch')
axs[2].set_ylabel('Number of batches')
plt.savefig('document_per_batch_hist_after.png')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(20, 7))
axs[0].hist(documents_lengths[subsets[0]])
axs[0].set_title(subsets[0])
axs[0].set_xlabel('Number of sentences in document')
axs[0].set_ylabel('Number of documents')
axs[1].hist(documents_lengths[subsets[1]])
axs[1].set_title(subsets[1])
axs[1].set_xlabel('Number of sentences in document')
axs[1].set_ylabel('Number of documents')
axs[2].hist(documents_lengths[subsets[2]])
axs[2].set_title(subsets[2])
axs[2].set_xlabel('Number of sentences in document')
axs[2].set_ylabel('Number of documents')
#plt.savefig('length_per_document_hist.png')
plt.show()

print(batch_length_1_number)