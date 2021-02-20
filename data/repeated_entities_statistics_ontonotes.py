import collections


def read_data(filename):
    rows = open(filename, 'r').read().strip().split('\n\n')
    documents, documents_tags, sentences, sentences_tags = [], [], [], []

    for document in rows:
        words = []
        tags = []
        words_and_tags = [line.split() for line in document.splitlines()]
        for word_and_tag in words_and_tags:
            if word_and_tag != []:
                words.append(word_and_tag[0])
                tags.append(word_and_tag[1])
        sentence = []
        sentence_tags = []
        for idx in range(len(words)):
            if words[idx] in ['.', '?', '!']:
                sentence.append(words[idx])
                sentence = ['<START>'] + sentence + ['<END>']
                sentence_tags.append(tags[idx])
                sentence_tags = ['NONE'] + sentence_tags + ['NONE']
                sentences += sentence
                sentences_tags += sentence_tags
                sentence = []
                sentence_tags = []
            else:
                sentence.append(words[idx])
                sentence_tags.append(tags[idx])
        documents.append(sentences)
        documents_tags.append(sentences_tags)
        sentences = []
        sentences_tags = []

    return documents, documents_tags

def get_documents_entities(document, document_tags):
    counter = collections.Counter()

    entities = []
    entities_tags = []
    for idx in range(len(document_tags)):

        if document_tags[idx][0] == 'B' or document_tags[idx][0] == 'I':
            entities.append([idx, document[idx]])
            entities_tags.append([idx, document_tags[idx]])
            counter[document[idx]] += 1

    return entities, entities_tags, counter

def make_sentences_mask(documents, documents_tags):
    # make a mask for repeated entities in each document
    sentences = []
    tags = []
    masks = []

    for document, document_tags in zip(documents, documents_tags):
        sentence = []
        sentence_tags = []
        sentence_mask = []

        _, _, document_entities_counter = get_documents_entities(document, document_tags)
        #print(document_entities_counter)
        repeated_entities = {}

        for key in document_entities_counter.keys():
            if document_entities_counter[key] >= 2:
                repeated_entities[key] = document_entities_counter[key]

        repeated_entities = set(repeated_entities.keys())
        #print(repeated_entities)

        #break
        for idx in range(len(document)):
            if document[idx] == '<START>':
                sentence = []
                sentence_tags = []
                sentence_mask = []
            elif document[idx] == '<END>':
                sentences.append(sentence)
                tags.append(sentence_tags)
                masks.append(sentence_mask)
            else:
                sentence.append(document[idx])
                sentence_tags.append(document_tags[idx])
                if repeated_entities:
                    if document[idx] in repeated_entities:
                        if document_tags[idx] != 'O':
                            sentence_mask.append(1)
                        else:
                            sentence_mask.append(0)
                    else:
                        sentence_mask.append(0)

    return sentences, tags, masks


if __name__ == '__main__':
    # отдает документы
    # почистить знаки препинания /знак препинания -> знак препинания (пример: /. -> .)
    documents, documents_tags = read_data('ontonotes/v3/onto.train.ner')
    sentences, tags, masks = make_sentences_mask(documents, documents_tags)

    for i in range(10):
        print(sentences[i])
        print(tags[i])
        print(masks[i])
        print('-' * 10)
