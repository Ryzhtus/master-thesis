import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from named_entity_recognition.reader import ReaderCoNLL, ReaderOntonotes
from named_entity_recognition.reader_document import ReaderDocumentCoNLL

class DatasetNER(Dataset):
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


class DatasetDocumentNER(Dataset):
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

        return tokens_ids, tags_ids, tokenized_mask


class DocumentBatchIterator():
    def __init__(self, dataset, document2sentences):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.document2sentences = document2sentences
        self.batches_count = len(document2sentences.keys())

    def __len__(self):
        return self.batches_count

    def __iter__(self):
        return self._iterate_batches()

    def _iterate_batches(self):
        for document_id in range(self.batches_count):

            document_sentences_ids = self.document2sentences[document_id]

            batch_tokens_ids = []
            batch_tags_ids = []
            batch_tokenized_mask = []

            for sentence_id in document_sentences_ids:
                sentence_token_ids, sentence_tag_ids, sentence_mask = self.dataset[sentence_id]
                batch_tokens_ids.append(sentence_token_ids)
                batch_tags_ids.append(sentence_tag_ids)
                batch_tokenized_mask.append(sentence_mask)

            max_sentence_length = len(max(batch_tokens_ids, key=len))

            for batch_element_id in range(len(batch_tokens_ids)):
                if len(batch_tokens_ids[batch_element_id]) < max_sentence_length:
                    for i in range(len(batch_tokens_ids[batch_element_id]), max_sentence_length):
                        batch_tokens_ids[batch_element_id].append(0)
                        batch_tags_ids[batch_element_id].append(0)
                        batch_tokenized_mask[batch_element_id].append(0)

            yield [
                torch.LongTensor(batch_tokens_ids),
                torch.LongTensor(batch_tags_ids),
                torch.LongTensor(batch_tokenized_mask)
            ]

def create_dataset_and_dataloader(dataset_name, filename, batch_size, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = DatasetNER(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=True, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = DatasetNER(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=True, collate_fn=dataset.paddings)

def create_dataset_and_document_level_iterator(dataset_name, filename, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderDocumentCoNLL()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = DatasetDocumentNER(sentences, tags, masks, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences)

        return dataset, data_iterator