from typing import List

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer


class Document(Dataset):
    def __init__(self, sentences: List[int],
                       document2sentences: dict,
                       tokenizer: PreTrainedTokenizer,
                       max_sequence_length: int = 128):

        self.sentences = sentences
        self.document2sentences = document2sentences
        self.max_sequence_length = max_sequence_length
        self.documents = [[self.sentences[sentence_id] for sentence_id in self.document2sentences[document_id]]
                          for document_id in self.document2sentences.keys()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        """
        Returns all sentences of the particular document encoded by a tokenizer
        """
        document = self.documents[item]
        document_ids = []

        for sentence in document:
            tokens = []
            for word in sentence:
                subtokens = self.tokenizer.tokenize(word)
                tokens.extend(subtokens)

            document_ids.append(tokens)

        # max_length = len(max(document_ids, key=lambda x: len(x)))

        for sentence_id, tokens in enumerate(document_ids):
            if len(tokens) < self.max_sequence_length:
                difference = self.max_sequence_length - len(tokens)
                tokens += [self.tokenizer.pad_token] * (difference - 2)

            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            document_ids[sentence_id] = torch.LongTensor(tokens_ids).unsqueeze(0)

        return torch.LongTensor(torch.cat(document_ids, dim=0))

    def get_document_tokens(self, item):
        """
        Returns all tokenized sentences of the particular document, but not encoded
        """
        document = self.documents[item]
        document_bpe = []

        for sentence in document:
            tokens = []
            for word in sentence:
                subtokens = self.tokenizer.tokenize(word)
                tokens.extend(subtokens)

            document_bpe.append(tokens)

        max_length = len(max(document_bpe, key=lambda x: len(x)))

        for sentence_id, tokens in enumerate(document_bpe):
            if len(tokens) < max_length:
                difference = max_length - len(tokens)
                tokens += [self.tokenizer.pad_token] * difference

            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            document_bpe[sentence_id] = tokens

        return document_bpe

    def find_token_positions_for_each_word(self, tokens):
        """
        Returns a dict, where keys are words and values are lists with corresponding positions of tokens forming the word
        {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [6], ...}
        """
        words_ids = {}
        current_word_ids = []
        current_word_bpe = []
        word_id = 0

        for idx in range(len(tokens) - 1):
            if ('##' not in tokens[idx]) and ('##' not in tokens[idx + 1]):
                current_word_ids.append(idx)
                current_word_bpe.append(tokens[idx])
                words_ids[word_id] = {'bpe': current_word_bpe, 'positions': current_word_ids}
                word_id += 1
                current_word_bpe = []
                current_word_ids = []
            elif ('##' in tokens[idx]) and ('##' not in tokens[idx + 1]):
                current_word_ids.append(idx)
                current_word_bpe.append(tokens[idx])
                words_ids[word_id] = {'bpe': current_word_bpe, 'positions': current_word_ids}
                word_id += 1
                current_word_bpe = []
                current_word_ids = []
            else:
                current_word_ids.append(idx)
                current_word_bpe.append(tokens[idx])

        # append last [SEP] token
        current_word_ids.append(len(tokens) - 1)
        current_word_bpe.append(tokens[-1])
        words_ids[word_id] = {'bpe': current_word_bpe, 'positions': current_word_ids}

        return words_ids

    def collect_all_positions_for_each_word(self, item):
        """
        Returns a dict with all words in a document with their bpe-tokens positions in each sentence of the document
        Format: {Word Id (key): {tokens: [list of bpe-tokens of a particular word], pos: [list of dicts with sentence_id
        and positions of bpe-tokens in the corresponding sentence]} }
        """
        document = self.get_document_tokens(item)

        word_id = 0
        words = {}
        is_in_words = False

        for sentence_id, sentence in enumerate(document):
            sentence_words = self.find_token_positions_for_each_word(sentence)
            for word in sentence_words:
                for key in words:
                    if words[key]['bpe'] == sentence_words[word]['bpe']:
                        words[key]['pos'].append({'sentence_id': sentence_id, 'ids': sentence_words[word]['positions']})
                        is_in_words = True

                if is_in_words:
                    is_in_words = False
                else:
                    words[word_id] = {'bpe': sentence_words[word]['bpe'],
                                      'pos': [{'sentence_id': sentence_id, 'ids': sentence_words[word]['positions']}]}
                    word_id += 1

        return words

    def get_document_words_by_sentences(self, item):
        document = self.get_document_tokens(item)

        document_words_by_sentences = {}

        for sentence_id, sentence in enumerate(document):
            document_words_by_sentences[sentence_id] = self.find_token_positions_for_each_word(sentence)

        return document_words_by_sentences