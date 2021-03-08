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
    document_tags = [tags for tags in document_tags if tags != []]
    return document, document_tags

def get_sentences(path):
    documents = []
    documents_tags = []

    for _, _, files in os.walk(path):
        for filename in files:
            sentences, sentences_tags = read_document(path + '/' + filename)

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

    return counter

def make_sentence_mask(document, counter):
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

def print_statistics(subset):
    paths = {'train': 'ontonotes/train', 'dev': 'ontonotes/development', 'test': 'ontonotes/test'}
    path = paths[subset]

    entities_number = 0
    repeated_entities_number = 0
    repeated_entities_set = set()
    documents_number = 0
    documents_with_repeated_entities_number = 0

    for _, _, files in os.walk(path):
        for filename in files:
            document, document_tags = read_document(path + '/' + filename)
            document_entities_counter = get_documents_entities(document, document_tags)
            documents_number += 1
            repeated_entities = {}
            for key in document_entities_counter.keys():
                if document_entities_counter[key] >= 2:
                    repeated_entities[key] = document_entities_counter[key]
            entities_number += sum(document_entities_counter.values())
            if repeated_entities:
                documents_with_repeated_entities_number += 1
                repeated_entities_set.update(set(repeated_entities.keys()))
                repeated_entities_number += sum(repeated_entities.values())

    print('Subset:', subset)
    print('Total number of documents                       : {}'.format(documents_number))
    print('Total number of documents with repeated entities: {}'.format(documents_with_repeated_entities_number))
    print('Total number of entities                        : {}'.format(entities_number))
    print('Total number of repeated entities in the text   : {}'.format(repeated_entities_number))
    print('Total number of unique repeated entities        : {}'.format(len(repeated_entities_set)))
    print('Repeated entities ratio                         : {:.2%}'.format(repeated_entities_number / entities_number))
    print()



if __name__ == '__main__':
    # documents, documents_tags = get_sentences('ontonotes/train')
    print_statistics('train')
    print_statistics('dev')
    print_statistics('test')
