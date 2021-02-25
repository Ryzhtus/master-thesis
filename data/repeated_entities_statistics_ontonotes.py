import collections
import os

def read_document(filename):
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
    return document, document_tags

def get_sentences(path):
    documents = []
    documents_tags = []

    for _, _, files in os.walk(path):
        for filename in files:
            print(filename)
            sentences, sentences_tags = read_document(path + '/' +filename)

            documents += sentences
            documents_tags += sentences_tags

    return documents, documents_tags

def get_documents_entities(document, document_tags):
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
            print(sentence)
            print(tags)
            print(sentences_entities)
            print('---------------------')
    return counter

if __name__ == '__main__':
    documents, documents_tags = get_sentences('ontonotes/train')

