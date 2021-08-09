from typing import List, Dict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers import PreTrainedTokenizer

class CoNLLDatasetBERT(Dataset):
    def __init__(self, sentences, tags, repeated_entities_masks, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags
        self.repeated_entities_masks = repeated_entities_masks

        self.tokenizer = tokenizer

        self.ner_tags = [self.tokenizer.pad_token] + list(
            set(tag for tag_list in self.sentences_tags for tag in tag_list))
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
        words_ids = []
        tokenized_tags = []
        tokenized_mask = []

        for word_id, word in enumerate(words):
            subtokens = self.tokenizer.tokenize(word)
            for i in range(len(subtokens)):
                if i == 0:
                    tokenized_tags.append(word2tag[word])
                    tokenized_mask.append(word2mask[word])
                else:
                    tokenized_tags.append(-100)
                    tokenized_mask.append(word2mask[word])

                words_ids.append(word_id)
            tokens.extend(subtokens)

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tags_ids = [-100] + [self.tag2idx[tag] if tag != -100 else tag for tag in tokenized_tags] + [-100]

        tokenized_mask = [-1] + tokenized_mask + [-1]
        attention_mask = [1 for token in tokens_ids]

        return torch.LongTensor(tokens_ids), torch.LongTensor(tags_ids), torch.LongTensor(tokenized_mask), \
               torch.LongTensor(attention_mask), words_ids

    def paddings(self, batch):
        tokens, tags, masks, attention_masks, words_ids= list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        tags = pad_sequence(tags, batch_first=True, padding_value=-100)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return tokens, tags, masks, attention_masks, words_ids


class CoNLLDatasetT5(Dataset):
    def __init__(self, sentences: List[List[str]], labels: List[List[str]], tokenizer):
        self.sentences = sentences
        self.labels = labels

        self.tokenizer = tokenizer

        self.entity_tags = sorted(list(set(tag for tag_list in self.labels for tag in tag_list)))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.entity_tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.entity_tags)}

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = self.sentences[item]
        labels = self.labels[item]

        word2tag = dict(zip(words, labels))

        tokens = []
        words_ids = []
        tokenized_labels = []

        for word_id, word in enumerate(words):
            subtokens = self.tokenizer.tokenize(word)
            for i in range(len(subtokens)):
                tokenized_labels.append(word2tag[word])
                words_ids.append(word_id)

            tokens.extend(subtokens)

        tokens = tokens + [self.tokenizer.eos_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        label_ids = [self.tag2idx[label] for label in tokenized_labels] + [-100]

        attention_mask = [1 for _ in tokens_ids]

        return torch.LongTensor(tokens_ids), torch.LongTensor(label_ids), torch.LongTensor(attention_mask), words_ids

    def paddings(self, batch):
        tokens, labels, attention_masks, words_ids = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return tokens, labels, attention_masks, words_ids


class SentencesPlusDocumentsDataset(Dataset):
    def __init__(self, sentences: List[List[str]], labels: List[List[str]],
                 document2sentences: Dict,  sentence2position: Dict, tokenizer: PreTrainedTokenizer):
        self.sentences = sentences
        self.labels = labels

        self.entity_tags = sorted(list(set(tag for tag_list in self.labels for tag in tag_list)))
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.entity_tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.entity_tags)}

        # словарь, где по номеру документа получаем номера всех его предложений относительно корпуса
        self.document2sentences = document2sentences
        # словарь, в котором по общему номеру предложения (относительно корпуса) можно получить номер документа,
        # которому он принадлежит и его порядковый номер в этом документе
        self.sentence2position = sentence2position
        # словарь, где по номеру предложения получаем номер документа, в котором оно находится
        self.sentence2document = {sentence_id: document_id for document_id in self.document2sentences.keys()
                                  for sentence_id in self.document2sentences[document_id]}

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        words = self.sentences[item]
        labels = self.labels[item]
        document_id = self.sentence2document[item]
        sentence_position_in_document = self.sentence2position[item]['sentence_pos_id']

        word2tag = dict(zip(words, labels))

        tokens = []
        words_ids = []
        tokenized_labels = []

        for word_id, word in enumerate(words):
            subtokens = self.tokenizer.tokenize(word)
            for i in range(len(subtokens)):
                tokenized_labels.append(word2tag[word])

            words_ids.append(len(tokens))
            tokens.extend(subtokens)

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        label_ids = [-100] + [self.tag2idx[label] for label in tokenized_labels] + [-100]

        attention_mask = [1 for _ in tokens_ids]

        return torch.LongTensor(tokens_ids), torch.LongTensor(label_ids), torch.LongTensor(attention_mask), \
               words_ids, document_id, sentence_position_in_document

    def paddings(self, batch):
        tokens, labels, attention_masks, words_ids, document_ids, sentences_ids = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return tokens, labels, attention_masks, words_ids, document_ids, sentences_ids


class SentencesDataset(Dataset):
    def __init__(self, sentences, tags, repeated_entities_masks, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags
        self.repeated_entities_masks = repeated_entities_masks

        self.tokenizer = tokenizer

        self.ner_tags = [self.tokenizer.pad_token] + list(
            set(tag for tag_list in self.sentences_tags for tag in tag_list))
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
            subtokens = self.tokenizer.tokenize(word)
            for i in range(len(subtokens)):
                tokenized_tags.append(word2tag[word])
                tokenized_mask.append(word2mask[word])
            tokens.extend(subtokens)

        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokenized_tags = ['O'] + tokenized_tags + ['O']
        tags_ids = [self.tag2idx[tag] for tag in tokenized_tags]

        tokenized_mask = [-1] + tokenized_mask + [-1]

        return tokens_ids, tags_ids, tokenized_mask
