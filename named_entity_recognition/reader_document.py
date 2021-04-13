import collections
import os


class ReaderDocumentCoNLL():
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

    def get_sentences(self, filename):
        output_documents = []
        output_documents_tags = []
        documents_masks = []

        sentences, sentences_tags = self.read_document(filename)
        documents, documents_tags = self.convert_to_document(sentences, sentences_tags)

        document_id2sentences_ids = {}
        documents = [document for document in documents if document != []]

        sentence_id = 0
        for document_id, document in enumerate(documents):
            document_id2sentences_ids[document_id] = []
            for sentence in document:
                document_id2sentences_ids[document_id].append(sentence_id)
                sentence_id += 1

        documents_tags = [tags for tags in documents_tags if tags != []]

        for document, document_tags in zip(documents, documents_tags):
            document_entities_counter = self.get_documents_entities(document, document_tags)
            repeated_entities = {}
            for key in document_entities_counter.keys():
                if document_entities_counter[key] >= 2:
                    repeated_entities[key] = document_entities_counter[key]
            documents_masks += self.make_sentence_mask(document, repeated_entities)

            output_documents += document
            output_documents_tags += document_tags

        return output_documents, output_documents_tags, documents_masks, document_id2sentences_ids

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