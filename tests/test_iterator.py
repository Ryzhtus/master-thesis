import unittest

from named_entity_recognition.iterator import DocumentBatchIterator
from named_entity_recognition.dataset import DatasetDocumentNER
from named_entity_recognition.reader import ReaderDocumentCoNLL

from transformers import BertTokenizer

class DocumentIteratorTest(unittest.TestCase):
    def test_batches(self):
        paths = ['../data/conll2003/train.txt', '../data/conll2003/valid.txt', '../data/conll2003/test.txt']
        for filepath in paths:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
            reader = ReaderDocumentCoNLL()
            sentences, tags, masks, document2sentences = reader.get_sentences(filepath)
            dataset = DatasetDocumentNER(sentences, tags, masks, tokenizer)
            data_iterator = DocumentBatchIterator(dataset, document2sentences, group_documents=True,
                                                  batch_size=32, shuffle=False)

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


if __name__ == '__main__':
    unittest.main()
