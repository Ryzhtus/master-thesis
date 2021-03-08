import collections
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from named_entity_recognition.reader import ReaderCoNLL, ReaderOntonotes

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


def create_dataset_and_dataloader(dataset_name, filename, batch_size, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL()
        sentences, sentences_tags = reader.read_document(path)
        documents, documents_tags = reader.convert_to_document(sentences, sentences_tags)
        sentences, tags, masks = reader.make_sentences_mask(documents)
        dataset = DatasetNER(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, num_workers=4, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = DatasetNER(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, num_workers=4, collate_fn=dataset.paddings)