import collections


class ReaderCoNLL():
    def __init__(self, include_document_ids=False):
        self.include_document_ids = include_document_ids

    @staticmethod
    def __read_document(filename: str):
        rows = open(filename, 'r').read().strip().split("\n\n")
        sentences, sentences_tags = [], []

        for sentence in rows:
            words = [line.split()[0] for line in sentence.splitlines()]
            tags = [line.split()[-1] for line in sentence.splitlines()]
            sentences.append(words)
            sentences_tags.append(tags)

        return sentences, sentences_tags

    @staticmethod
    def __convert_to_document(sentences: list, tags: list):
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

    @staticmethod
    def __get_documents_entities(document: list, document_tags: list):
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

    @staticmethod
    def __make_sentence_mask(document: list, counter: dict):
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
