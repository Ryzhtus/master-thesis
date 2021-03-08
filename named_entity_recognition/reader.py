import collections
import os


class ReaderCoNLL():
    def read_document(self, filename):
        rows = open(filename, 'r').read().strip().split("\n\n")
        sentences, sentences_tags = [], []

        for sentence in rows:
            words = [line.split()[0] for line in sentence.splitlines()]
            tags = [line.split()[-1] for line in sentence.splitlines()]
            sentences.append(words)
            sentences_tags.append(tags)

        return sentences, sentences_tags

    def convert_to_document(self, sentences, tags):
        documents = []
        documents_tags = []
        document = []
        document_tags = []

        for sentence, tag in zip(sentences, tags):
            if '-DOCSTART-' in sentence:
                documents.append(document)
                documents_tags.append(document_tags)
                document = []
                document_tags = []
            else:
                document.append(sentence)
                document_tags.append(tag)

        # append last document, because there is no '-DOCSTART-' or special end marker in text further
        documents.append(document)
        documents_tags.append(document_tags)

        return documents, documents_tags

    def get_sentences(self, path):
        output_documents = []
        output_documents_tags = []
        documents_masks = []

        sentences, sentences_tags = self.read_document(path)
        documents, documents_tags = self.convert_to_document(sentences, sentences_tags)

        for document, document_tags in zip(documents, documents_tags):
            document_entities_counter = self.get_documents_entities(document, document_tags)
            repeated_entities = {}
            for key in document_entities_counter.keys():
                if document_entities_counter[key] >= 2:
                    repeated_entities[key] = document_entities_counter[key]

            documents_masks += self.make_sentence_mask(document, repeated_entities)

        output_documents += sentences
        output_documents_tags += sentences_tags

        return sentences, sentences_tags, documents_masks

    def get_documents_entities(self, document, document_tags):
        counter = collections.Counter()

        for sentence, tags in zip(document, document_tags):
            sentence_entity = []
            sentences_entities = []
            entity = []
            entity_tags = []
            entity_ids = []
            for idx in range(len(sentence)):
                if tags[idx] != 'O':
                    entity.append(sentence[idx])
                    entity_tags.append(tags[idx])
                    entity_ids.append(idx)

            if entity:
                sentence_entity.append(entity[0])
                for idx in range(1, len(entity)):
                    if entity_tags[idx][0] == 'B':
                        sentences_entities.append(sentence_entity)
                        sentence_entity = [entity[idx]]
                    else:
                        sentence_entity.append(entity[idx])
                sentences_entities.append(sentence_entity)

                sentences_entities = [' '.join(entity) for entity in sentences_entities]
                for entity in sentences_entities:
                    counter[entity] += 1

        return counter

    def make_sentence_mask(self, document, counter):
        masks = []
        for sentence in document:
            sentence_mask = [-1 for x in range(len(sentence))]
            for key in list(counter.keys()):
                entity = key
                window_size = len(entity.split(' '))
                for window_start in range(0, len(sentence) - window_size):
                    if ' '.join(sentence[window_start: window_start + window_size]) == entity:
                        for idx in range(window_start, window_start + window_size):
                            sentence_mask[idx] = 1
            masks.append(sentence_mask)

        return masks


class ReaderOntonotes():
    def read_document(self, filename):
        rows = open(filename, 'r').read().strip().split('\n')
        document, document_tags = [], []
        sentence, tags = [], []

        for row in rows:
            data = row.split(' ')

            # check if sentence is emtpy
            if data[0] == '':
                document.append(sentence)
                document_tags.append(tags)
                sentence = []
                tags = []
            else:
                sentence.append(data[0])
                tags.append(data[1])
        document = [sentence for sentence in document if sentence != []]
        document_tags = [tags for tags in document_tags if tags != []]

        document_entities_counter = self.get_documents_entities(document, document_tags)
        repeated_entities = {}

        for key in document_entities_counter.keys():
            if document_entities_counter[key] >= 2:
                repeated_entities[key] = document_entities_counter[key]

        document_masks = self.make_sentence_mask(document, repeated_entities)

        return document, document_tags, document_masks

    def get_sentences(self, path):
        documents = []
        documents_tags = []
        documents_masks = []

        for _, _, files in os.walk(path):
            for filename in files:
                sentences, sentences_tags, sentences_masks = self.read_document(path + '/' + filename)

                documents += sentences
                documents_tags += sentences_tags
                documents_masks += sentences_masks

        return documents, documents_tags, documents_masks

    def get_documents_entities(self, document, document_tags):
        counter = collections.Counter()
        for sentence, tags in zip(document, document_tags):
            sentence_entity = []
            sentences_entities = []
            entity = []
            entity_tags = []
            entity_ids = []
            for idx in range(len(sentence)):
                if tags[idx] != 'O':
                    entity.append(sentence[idx])
                    entity_tags.append(tags[idx])
                    entity_ids.append(idx)

            if entity:
                sentence_entity.append(entity[0])
                for idx in range(1, len(entity)):
                    if entity_tags[idx][0] == 'B':
                        sentences_entities.append(sentence_entity)
                        sentence_entity = [entity[idx]]
                    else:
                        sentence_entity.append(entity[idx])
                sentences_entities.append(sentence_entity)

                sentences_entities = [' '.join(entity) for entity in sentences_entities]
                for entity in sentences_entities:
                    counter[entity] += 1

        return counter

    def make_sentence_mask(self, document, counter):
        masks = []
        for sentence in document:
            sentence_mask = [0 for x in range(len(sentence))]
            for key in list(counter.keys()):
                entity = key
                window_size = len(entity.split(' '))
                for window_start in range(0, len(sentence) - window_size):
                    if ' '.join(sentence[window_start: window_start + window_size]) == entity:
                        for idx in range(window_start, window_start + window_size):
                            sentence_mask[idx] = 1
            masks.append(sentence_mask)

        return masks


if __name__ == '__main__':
    """reader = ReaderOntonotes()
    documents, documents_tags, documents_masks = reader.get_sentences('../data/ontonotes/development')
    for i in range(10):
        print(len(documents[i]), documents[i])
        print(len(documents_tags[i]), documents_tags[i])
        print(len(documents_masks[i]), documents_masks[i])"""

    reader = ReaderCoNLL()
    documents, documents_tags, documents_masks = reader.get_sentences('../data/conll2003/train.txt')
    for i in range(10):
        print(len(documents[i]), documents[i])
        print(len(documents_tags[i]), documents_tags[i])
        print(len(documents_masks[i + 1]), documents_masks[i + 1])
