# This script makes a collection of entities that occurs twice or more times in the particular document in CoNLL2003
import collections


def read_document(filename):
    rows = open(filename, 'r').read().strip().split("\n\n")
    sentences, sentences_tags = [], []

    for sentence in rows:
        words = [line.split()[0] for line in sentence.splitlines()]
        tags = [line.split()[-1] for line in sentence.splitlines()]
        sentences.append(words)
        sentences_tags.append(tags)

    return sentences, sentences_tags

def convert_to_document(sentences, tags):
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

def print_statistics(subset):
    paths = {'train': 'conll2003/train.txt', 'dev': 'conll2003/valid.txt', 'test': 'conll2003/test.txt'}
    path = paths[subset]

    entities_number = 0
    repeated_entities_number = 0
    repeated_entities_set = set()
    documents_number = 0
    documents_with_repeated_entities_number = 0

    sentences, sentences_tags = read_document(path)
    documents, documents_tags = convert_to_document(sentences, sentences_tags)

    documents = [document for document in documents if document != []]
    documents_tags = [tags for tags in documents_tags if tags != []]

    for document, document_tags in zip(documents, documents_tags):
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
    print_statistics('train')
    print_statistics('dev')
    print_statistics('test')



