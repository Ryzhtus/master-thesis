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

def make_sentences_mask(documents):
    # make a mask for repeated entities in each document
    sentences = []
    tags = []
    masks = []

    for document in documents:
        sentence = []
        sentence_tags = []
        sentence_mask = []

        _, _, document_entities_counter = get_documents_entities(document)
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
                        sentence_mask.append(0)
                else:
                    sentence_mask.append(0)

    return sentences, tags, masks


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


def print_example(subset, id):
    file_paths = {'train': 'conll2003/train.txt', 'eval': 'conll2003/valid.txt', 'test': 'conll2003/test.txt'}
    sentences, tags, tags_number = read_data(file_paths[subset])
    print('Subset:', subset)
    print('Sentences size:', len(sentences), 'Tags size:', len(tags), 'Tags number:', tags_number)
    documents = convert_to_document(sentences, tags)
    sentences, tags, masks = make_sentences_mask(documents)
    print()
    print('After processing:')
    print('Sentences size:', len(sentences), 'Tags size:', len(tags))
    print()
    print('Example:')
    print('Sentence ID={}: length: {}, values: {}'.format(id, len(sentences[id]), sentences[id]))
    print('Tags     ID={}: length: {}, values: {}'.format(id, len(tags[id]), tags[id]))
    print('Mask     ID={}: length: {}, values: {}'.format(id, len(masks[id]), masks[id]))


if __name__ == '__main__':
    print_statistics()
    find_repeated_entities('train')
    find_repeated_entities('eval')
    find_repeated_entities('test')
    print_example('train', 0)


"""
OUTPUT: 

Amount of documents for each CoNLL subset:
Train: 945
Eval : 215
Test : 230

Subset: train
Total number of documents with repeated entities: 873
Total number of unique repeated entities        : 2676
Total number of repeated entities in the text   : 16160

Subset: eval
Total number of documents with repeated entities: 191
Total number of unique repeated entities        : 953
Total number of repeated entities in the text   : 4120

Subset: test
Total number of documents with repeated entities: 202
Total number of unique repeated entities        : 875
Total number of repeated entities in the text   : 3724

Subset: train
Sentences size: 14987 Tags size: 14987 Tags number: 204567

After processing:
Sentences size: 14042 Tags size: 14042

Example:
Sentence ID=100: length: 9, values: ['Rabinovich', 'is', 'winding', 'up', 'his', 'term', 'as', 'ambassador', '.']
Tags     ID=100: length: 9, values: ['B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
Mask     ID=100: length: 9, values: [1, 0, 0, 0, 0, 0, 0, 0, 0]
"""

