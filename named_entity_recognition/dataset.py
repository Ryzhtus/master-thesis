import collections
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CoNLL2003Dataset(Dataset):
    def __init__(self, sentences, tags, repeated_entities_masks, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags
        self.repeated_entities_masks = repeated_entities_masks

        self.tokenizer = tokenizer

        self.ner_tags = ['<PAD>'] + list(set(tag for tag_list in self.sentences_tags for tag in tag_list))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.ner_tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.ner_tags)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = self.sentences[item]
        tags = self.sentences_tags[item]
        mask = self.repeated_entities_masks[item]

        word2mask = dict(zip(words, mask))
        word2tag = dict(zip(words, tags))

        tokens = []
        tokenized_tags = []
        tokenized_mask = []

        for word in words:
            if word not in ('[CLS]', '[SEP]'):
                subtokens = self.tokenizer.tokenize(word)
                for i in range(len(subtokens)):
                    tokenized_tags.append(word2tag[word])
                    tokenized_mask.append(word2mask[word])
                tokens.extend(subtokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokenized_tags = ['O'] + tokenized_tags + ['O']
        tags_ids = [self.tag2idx[tag] for tag in tokenized_tags]

        tokenized_mask = [-1] + tokenized_mask + [-1]

        return torch.LongTensor(tokens_ids), torch.LongTensor(tags_ids), torch.LongTensor(tokenized_mask)

    def paddings(self, batch):
        tokens, tags, masks = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True)
        tags = pad_sequence(tags, batch_first=True)
        masks = pad_sequence(masks, batch_first=True)

        return tokens, tags, masks


def read_data(filename):
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
                        sentence_mask.append(-1)
                else:
                    sentence_mask.append(-1)

    return sentences, tags, masks


def create_dataset_and_dataloader(filename, batch_size, tokenizer):
    sentences, tags = read_data(filename)
    documents = convert_to_document(sentences, tags)
    sentences, tags, masks = make_sentences_mask(documents)
    dataset = CoNLL2003Dataset(sentences, tags, masks, tokenizer)

    return dataset, DataLoader(dataset, batch_size, num_workers=4, collate_fn=dataset.paddings)