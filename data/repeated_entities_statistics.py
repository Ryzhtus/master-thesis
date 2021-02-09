# This script makes a collection of entities that occurs twice or more times in the particular document in CoNLL2003
import collections


def read_data(filename):
    rows = open(filename, 'r').read().strip().split("\n\n")
    sentences, sentences_tags = [], []

    for sentence in rows:
        words = [line.split()[0] for line in sentence.splitlines()]
        tags = [line.split()[-1] for line in sentence.splitlines()]
        sentences.append(words)
        sentences_tags.append(tags)

    tags_number = sum([len(tag) for tag in sentences_tags])

    return sentences, sentences_tags, tags_number

def convert_to_document(sentences, tags):
    documents = []
    document = []
    document_tags = []

    for sentence, tag in zip(sentences, tags):

        if sentence == ['-DOCSTART-']:
            documents.append([document, document_tags])
            document = []
            document_tags = []
        else:
            document += sentence
            document_tags += tag

    return documents[1:]  # we do not include the first document because it's empty

def get_documents_entities(document):
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

def print_statistics():
    print('Amount of documents for each CoNLL subset:')
    train_sentences, train_tags, train_tags_number = read_data('conll2003/train.txt')
    train_documents = convert_to_document(train_sentences, train_tags)
    print('Train:', len(train_documents))

    eval_sentences, eval_tags, eval_tags_number = read_data('conll2003/valid.txt')
    eval_documents = convert_to_document(eval_sentences, eval_tags)
    print('Eval :', len(eval_documents))

    test_sentences, test_tags, test_tags_number = read_data('conll2003/test.txt')
    test_documents = convert_to_document(test_sentences, test_tags)
    print('Test :', len(test_documents))
    print()

def find_repeated_entities(subset, show_repeated_entities=False):
    file_paths = {'train': 'conll2003/train.txt', 'eval': 'conll2003/valid.txt', 'test': 'conll2003/test.txt'}
    sentences, tags, tags_number = read_data(file_paths[subset])
    train_documents = convert_to_document(sentences, tags)
    documents_number = 0
    entities_set = set()
    entities_sum = 0
    for document_id, document in enumerate(train_documents):
        document_entities, document_entities_tags, document_entities_counter = get_documents_entities(document)
        repeated_entities = {}
        for key in document_entities_counter.keys():
            if document_entities_counter[key] >= 2:
                repeated_entities[key] = document_entities_counter[key]
        if repeated_entities:
            entities_set.update(set(repeated_entities.keys()))
            entities_sum += sum(repeated_entities.values())
            if show_repeated_entities:
                print(document_id, dict(sorted(repeated_entities.items(), key=lambda item: item[1], reverse=True)))
            documents_number += 1

    print('Subset:', subset)
    print('Total number of documents with repeated entities:', documents_number)
    print('Total number of unique repeated entities        :', len(entities_set))
    print('Total number of repeated entities in the text   :', entities_sum)
    print()


if __name__ == '__main__':
    print_statistics()
    find_repeated_entities('train')
    find_repeated_entities('eval')
    find_repeated_entities('test')
