import torch
from torch.utils.data import Dataset

class Document(Dataset):
    def __init__(self, sentences, document2sentences, tokenizer):
        self.sentences = sentences
        self.document2sentences = document2sentences
        self.documents = [[self.sentences[sentence_id] for sentence_id in self.document2sentences[document_id]]
                          for document_id in self.document2sentences.keys()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        document = self.documents[item]
        document_ids = []

        for sentence in document:
            tokens = []
            for word in sentence:
                subtokens = self.tokenizer.tokenize(word)
                tokens.extend(subtokens)

            document_ids.append(tokens)

        max_length = len(max(document_ids, key=lambda x: len(x)))

        for sentence_id, tokens in enumerate(document_ids):
            if len(tokens) < max_length:
                difference = max_length - len(tokens)
                tokens += [self.tokenizer.pad_token] * difference

            tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            document_ids[sentence_id] = torch.LongTensor(tokens_ids).unsqueeze(0)

        return torch.LongTensor(torch.cat(document_ids, dim=0))

    def get_document(self, item):
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

    @staticmethod
    def find_token_positions_for_each_word(tokens):
        """
        Return Dict, where keys are words and values are lists with corresponding positions of tokens forming the word
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

    def collect_all_positions_for_each_token(self, item):
        document = self.get_document(item)

        word_id = 0
        words = {}
        is_in_words = False

        for sentence_id, sentence in enumerate(document):
            sentence_words = self.find_token_positions_for_each_word(sentence)
            for word in sentence_words:
                for key in words:
                    if words[key]['tokens'] == sentence_words[word]['bpe']:
                        words[key]['pos'].append({'sentence_id': sentence_id, 'ids': sentence_words[word]['positions']})
                        is_in_words = True

                if is_in_words:
                    is_in_words = False
                else:
                    words[word_id] = {'tokens': sentence_words[word]['bpe'],
                                      'pos': [{'sentence_id': sentence_id, 'ids': sentence_words[word]['positions']}]}
                    word_id += 1

        return words