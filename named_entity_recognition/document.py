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