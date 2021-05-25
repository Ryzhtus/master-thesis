import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from named_entity_recognition.reader import ReaderCoNLL, ReaderOntonotes, ReaderDocumentCoNLL, ReaderDocumentOntonotes
from named_entity_recognition.iterator import DocumentBatchIterator


class DatasetNER(Dataset):
    def __init__(self, sentences, tags, repeated_entities_masks, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags
        self.repeated_entities_masks = repeated_entities_masks

        self.tokenizer = tokenizer

        self.ner_tags = [self.tokenizer.pad_token] + list(set(tag for tag_list in self.sentences_tags for tag in tag_list))
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

        return torch.LongTensor(tokens_ids), torch.LongTensor(tags_ids), torch.LongTensor(tokenized_mask)

    def paddings(self, batch):
        tokens, tags, masks = list(zip(*batch))

        tokens = pad_sequence(tokens, batch_first=True)
        tags = pad_sequence(tags, batch_first=True)
        masks = pad_sequence(masks, batch_first=True)

        return tokens, tags, masks


class DatasetDocumentNER(Dataset):
    def __init__(self, sentences, tags, repeated_entities_masks, tokenizer):
        self.sentences = sentences
        self.sentences_tags = tags
        self.repeated_entities_masks = repeated_entities_masks

        self.tokenizer = tokenizer

        self.ner_tags = [self.tokenizer.pad_token] + list(set(tag for tag_list in self.sentences_tags for tag in tag_list))
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

            document_ids[sentence_id] = torch.LongTensor(tokens_ids)

        return torch.LongTensor(document_ids)


def create_dataset_and_dataloader(dataset_name: str, filename: str, batch_size: int, shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = DatasetNER(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = DatasetNER(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def create_dataset_and_document_level_iterator(dataset_name: str, filename: str, group_documents: bool, batch_size: int,
                                               tokenizer):
    if dataset_name == 'conll':
        reader = ReaderDocumentCoNLL()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = DatasetDocumentNER(sentences, tags, masks, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences, group_documents=group_documents,
                                              batch_size=batch_size, shuffle=True)

        return dataset, documents, data_iterator

    if dataset_name == 'ontonotes':
        reader = ReaderDocumentOntonotes()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = DatasetDocumentNER(sentences, tags, masks, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences, shuffle=True)

        return dataset, data_iterator
