import collections
import os


class ReaderCoNLL():
    def read(self, filename):
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
        document = []
        document_tags = []

        for sentence, tag in zip(sentences, tags):
            sentence = ['<START>'] + sentence + ['<END>']
            tag = ['NONE'] + tag + ['NONE']

            if '-DOCSTART-' in sentence:
                documents.append([document, document_tags])
                document = []
                document_tags = []
            else:
                document += sentence
                document_tags += tag

        # append last document, because there is no '-DOCSTART-' or special end marker in text further
        documents.append([document, document_tags])

        return documents

    def get_documents_entities(self, document):
        counter = collections.Counter()
        words = document[0]
        tags = document[1]

        entities = []
        entities_tags = []
        for idx in range(len(tags)):
            if tags[idx][0] == 'B' or tags[idx][0] == 'I':
                entities.append([idx, words[idx]])
                entities_tags.append([idx, tags[idx]])
                counter[words[idx]] += 1

        return entities, entities_tags, counter

    def make_sentences_mask(self, documents):
        # make a mask for repeated entities in each document
        sentences = []
        tags = []
        masks = []

        for document in documents:
            sentence = []
            sentence_tags = []
            sentence_mask = []

            _, _, document_entities_counter = self.get_documents_entities(document)
            repeated_entities = {}

            for key in document_entities_counter.keys():
                if document_entities_counter[key] >= 2:
                    repeated_entities[key] = document_entities_counter[key]

            repeated_entities = set(repeated_entities.keys())

            words = document[0]
            words_tags = document[1]

            for idx in range(len(words)):
                if words[idx] == '<START>':
                    sentence = []
                    sentence_tags = []
                    sentence_mask = []
                elif words[idx] == '<END>':
                    sentences.append(sentence)
                    tags.append(sentence_tags)
                    masks.append(sentence_mask)
                else:
                    sentence.append(words[idx])
                    sentence_tags.append(words_tags[idx])
                    if repeated_entities:
                        if words[idx] in repeated_entities:
                            sentence_mask.append(1)
                        else:
                            sentence_mask.append(-1)
                    else:
                        sentence_mask.append(-1)

        return sentences, tags, masks


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
        document_tags = [sentence for sentence in document if sentence != []]
        document_masks = [[0] * len(sentence) for sentence in document]
        return document, document_tags, document_masks

    def get_sentences(self, path):
        documents = []
        documents_tags = []
        documents_masks = []

        for _, _, files in os.walk(path):
            for filename in files:
                print(filename)
                sentences, sentences_tags, sentences_masks = self.read_document(path + '/' + filename)

                documents += sentences
                documents_tags += sentences_tags
                documents_masks += sentences_masks

        return documents, documents_tags, documents_masks


if __name__ == '__main__':
    reader = ReaderOntonotes()
    documents, documents_tags, documents_masks = reader.get_sentences('../data/ontonotes/train')
    #for i in range(10):
    #    print(len(documents[i]), documents[i], len(documents_masks[i]), documents_masks[i])