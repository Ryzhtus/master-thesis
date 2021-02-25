import unittest
from data.repeated_entities_statistics_ontonotes import read_document, make_sentences_mask


class RepeatedEntitiesTest(unittest.TestCase):
    def test_lengths_after_processing(self):
        paths = ['../data/ontonotes/onto.train.ner', '../data/ontonotes/onto.development.ner', '../data/ontonotes/onto.test.ner']
        for filepath in paths:
            documents, documents_tags = read_document(filepath)

            processed_sentences, processed_tags, masks = make_sentences_mask(documents, documents_tags)

            for sentence, tags, mask in zip(processed_sentences, processed_tags, masks):
                self.assertTrue(len(sentence) == len(tags) == len(mask))


if __name__ == '__main__':
    unittest.main()

