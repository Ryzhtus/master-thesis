import collections
import os

class ReaderCoNLL():
    def __init__(self, include_document_ids=False):
        self.include_document_ids = include_document_ids

    def __read_document(self, filename: str):
        rows = open(filename, 'r').read().strip().split("\n\n")
        sentences, sentences_tags = [], []

        for sentence in rows:
            words = [line.split()[0] for line in sentence.splitlines()]
            tags = [line.split()[-1] for line in sentence.splitlines()]
            sentences.append(words)
            sentences_tags.append(tags)

        return sentences, sentences_tags

    def __convert_to_document(self, sentences: list, tags: list):
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
                document.append(sentence)
                document_tags.append(tag)
            else:
                document.append(sentence)
                document_tags.append(tag)

        # append last document, because there is no '-DOCSTART-' or special end marker in text further
        documents.append(document)
        documents_tags.append(document_tags)

        return documents, documents_tags

    def __get_documents_entities(self, document: list, document_tags: list):
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

    def __make_sentence_mask(self, document: list, counter: dict):
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

    def __read_sentences(self, filename: str):
        output_documents = []
        output_documents_tags = []
        documents_masks = []

        sentences, sentences_tags = self.__read_document(filename)
        documents, documents_tags = self.__convert_to_document(sentences, sentences_tags)

        documents = [document for document in documents if document != []]
        documents_tags = [tags for tags in documents_tags if tags != []]

        for document, document_tags in zip(documents, documents_tags):
            document_entities_counter = self.__get_documents_entities(document, document_tags)
            repeated_entities = {}
            for key in document_entities_counter.keys():
                if document_entities_counter[key] >= 2:
                    repeated_entities[key] = document_entities_counter[key]
            documents_masks += self.__make_sentence_mask(document, repeated_entities)

            output_documents += document
            output_documents_tags += document_tags

        return output_documents, output_documents_tags, documents_masks

    def __read_documents(self, filename: str):
        output_documents = []
        output_documents_tags = []
        documents_masks = []

        sentences, sentences_tags = self.__read_document(filename)
        documents, documents_tags = self.__convert_to_document(sentences, sentences_tags)

        # dict {document_id: sentence_id}
        document_to_sentences = {}
        # dict {sentence_id: {document_id, sentence_position_in_document}}
        sentences_to_documents_to_positions = {}

        documents = [document for document in documents if document != []]

        sentence_id = 0
        for document_id, document in enumerate(documents):
            document_to_sentences[document_id] = []
            sentence_position_in_document = 0
            for sentence in document:
                document_to_sentences[document_id].append(sentence_id)
                sentences_to_documents_to_positions[sentence_id] = {'document_id': document_id,
                                                                    'sentence_pos_id': sentence_position_in_document}
                sentence_id += 1
                sentence_position_in_document += 1

        documents_tags = [tags for tags in documents_tags if tags != []]

        for document, document_tags in zip(documents, documents_tags):
            document_entities_counter = self.__get_documents_entities(document, document_tags)
            repeated_entities = {}
            for key in document_entities_counter.keys():
                if document_entities_counter[key] >= 2:
                    repeated_entities[key] = document_entities_counter[key]
            documents_masks += self.__make_sentence_mask(document, repeated_entities)

            output_documents += document
            output_documents_tags += document_tags

        return output_documents, output_documents_tags, documents_masks, document_to_sentences, sentences_to_documents_to_positions

    def read(self, filename: str):
        if self.include_document_ids:
            return self.__read_documents(filename)
        else:
            return self.__read_sentences(filename)


class ReaderOntonotes():
    def __init__(self, include_document_ids=False):
        self.include_document_ids = include_document_ids

    def __read_document(self, filename: str):
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

        document_entities_counter = self.__get_documents_entities(document, document_tags)
        repeated_entities = {}

        for key in document_entities_counter.keys():
            if document_entities_counter[key] >= 2:
                repeated_entities[key] = document_entities_counter[key]

        document_masks = self.__make_sentence_mask(document, repeated_entities)

        return document, document_tags, document_masks

    def __get_documents_entities(self, document: list, document_tags: list):
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

    def __make_sentence_mask(self, document: list, counter: dict):
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

    def __read_sentences(self, path: str):
        documents = []
        documents_tags = []
        documents_masks = []

        for _, _, files in os.walk(path):
            for filename in files:
                sentences, sentences_tags, sentences_masks = self.__read_document(path + '/' + filename)

                documents += sentences
                documents_tags += sentences_tags
                documents_masks += sentences_masks

        return documents, documents_tags, documents_masks

    def __read_documents(self, path: str):
        documents = []
        documents_tags = []
        documents_masks = []
        # dict {document_id: sentence_id}
        document_to_sentences = {}
        # dict {sentence_id: {document_id, sentence_position_in_document}}
        sentences_to_documents_to_positions = {}

        sentence_id = 0
        document_id = 0

        for _, _, files in os.walk(path):
            for filename in files:
                sentences, sentences_tags, sentences_masks = self.__read_document(path + '/' + filename)

                if sentences:
                    document_to_sentences[document_id] = []
                    sentence_position_in_document = 0
                    for sentence in sentences:
                        document_to_sentences[document_id].append(sentence_id)
                        sentences_to_documents_to_positions[sentence_id] = {'document_id': document_id,
                                                                            'sentence_pos_id': sentence_position_in_document}
                        sentence_position_in_document += 1
                        sentence_id += 1

                documents += sentences
                documents_tags += sentences_tags
                documents_masks += sentences_masks
                document_id += 1

        return documents, documents_tags, documents_masks, document_to_sentences, sentences_to_documents_to_positions

    def read(self, path: str):
        if self.include_document_ids:
            return self.__read_documents(path)
        else:
            return self.__read_sentences(path)