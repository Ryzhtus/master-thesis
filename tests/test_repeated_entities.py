import unittest
from data.repeated_entities_statistics import read_data, convert_to_document, make_sentences_mask


class RepeatedEntitiesTest(unittest.TestCase):
    def test_lengths_after_processing(self):
        paths = ['../data/conll2003/train.txt', '../data/conll2003/valid.txt', '../data/conll2003/test.txt']
        for filepath in paths:
            sentences, tags, tags_number = read_data(filepath)

            documents = convert_to_document(sentences, tags)
            processed_sentences, processed_tags, masks = make_sentences_mask(documents)

            for sentence, tags, mask in zip(processed_sentences, processed_tags, masks):
                self.assertTrue(len(sentence) == len(tags) == len(mask))

    def test_equal_sentences_lengths(self):
        paths = ['../data/conll2003/train.txt', '../data/conll2003/valid.txt', '../data/conll2003/test.txt']
        for filepath in paths:
            sentences, tags, tags_number = read_data(filepath)
            sentences_without_docstart_tag = []
            tags_without_docstart_tag = []

            for idx in range(len(sentences)):
                if sentences[idx] != ['-DOCSTART-']:
                    sentences_without_docstart_tag.append(sentences[idx])
                    tags_without_docstart_tag.append(tags[idx])

            documents = convert_to_document(sentences, tags)
            processed_sentences, processed_tags, _ = make_sentences_mask(documents)

            for original_sentence, processed_sentence in zip(sentences_without_docstart_tag, processed_sentences):
                self.assertEqual(original_sentence, processed_sentence)

            # self.assertEqual(len(sentences_without_docstart_tag), len(processed_sentences))
            # self.assertEqual(len(tags_without_docstart_tag), len(processed_tags))


if __name__ == '__main__':
    unittest.main()
