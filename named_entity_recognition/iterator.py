from itertools import chain, islice

import numpy
import torch


class DocumentBatchIterator():
    def __init__(self, dataset, document2sentences, group_documents, batch_size=20, shuffle=True):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.document2sentences = document2sentences
        self.sentence2document = {sentence_id: document_id for document_id in self.document2sentences.keys()
                                  for sentence_id in self.document2sentences[document_id] }
        self.batches_count = len(document2sentences.keys())
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.group_documents = group_documents

    def __len__(self):
        return self.batches_count

    def __iter__(self):
        if self.group_documents:
            return self._iterate_grouped_batches()
        else:
            return self._iterate_batches()

    def split_document(self, sentences_id):
        ids = sentences_id.copy()
        batches = []

        while ids:
            if len(ids) < self.batch_size:
                batches.append(ids)
                ids = []
            else:
                batch = []
                for idx in range(self.batch_size):
                    batch.append(ids.pop(0))

                batches.append(batch)

        return batches

    def grouper(self, n, li):
        it = chain(*li)

        out_l = []
        while True:
            chunk = list(islice(it, n))
            if len(chunk) < n:
                if chunk:
                    out_l[-1] += chunk
                return out_l
            out_l.append(chunk)

        return out_l

    def group_ids(self):
        if self.shuffle:
            document_ids = numpy.arange(self.batches_count)
            numpy.random.shuffle(document_ids)
        else:
            document_ids = numpy.arange(self.batches_count)

        batches_ids = []
        batch_ids = []
        for document_id in document_ids:
            if len(self.document2sentences[document_id]) >= self.batch_size:
                batches = self.split_document(self.document2sentences[document_id])

                for batch in batches:
                    if len(batch) == self.batch_size:
                        batches_ids.append(batch)
                    else:
                        batch_ids.append(batch)
            else:
                if len(batch_ids) > self.batch_size:
                    batches_ids.append(batch_ids)
                    batch_ids = []
                else:
                    batch_ids.append(self.document2sentences[document_id])

        batches_of_batch_size = []
        batches_for_following_group = []
        for batch in batches_ids:
            if isinstance(batch[0], list):
                for subbatch in batch:
                    if len(batch) == self.batch_size:
                        batches_of_batch_size.append(subbatch)
                    else:
                        batches_for_following_group.append(subbatch)
            else:
                if len(batch) == self.batch_size:
                    batches_of_batch_size.append(batch)
                else:
                    batches_for_following_group.append(batch)

        batches_of_batch_size += self.grouper(self.batch_size, batches_for_following_group)

        return batches_of_batch_size

    def _iterate_grouped_batches(self):
        batches = self.group_ids()
        self.batches_count = len(batches)

        for batch_ids in batches:
            batch_tokens_ids = []
            batch_tags_ids = []
            batch_tokenized_mask = []
            batch_documents = []

            for sentence_id in batch_ids:
                sentence_token_ids, sentence_tag_ids, sentence_mask = self.dataset[sentence_id]
                batch_tokens_ids.append(sentence_token_ids)
                batch_tags_ids.append(sentence_tag_ids)
                batch_tokenized_mask.append(sentence_mask)
                batch_documents.append(self.sentence2document[sentence_id])

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
                torch.LongTensor(batch_tokenized_mask),
                batch_documents
            ]

    def _iterate_batches(self):
        if self.shuffle:
            document_ids = numpy.arange(self.batches_count)
            numpy.random.shuffle(document_ids)
        else:
            document_ids = numpy.arange(self.batches_count)

        for document_id in document_ids:
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

class SentenceBatchIteratorWithDocumentInformation():
    def __init__(self, dataset, document2sentences, group_documents, batch_size=20, shuffle=True):
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.document2sentences = document2sentences
        self.batches_count = len(document2sentences.keys())
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return self.batches_count

    def __iter__(self):
        return self._iterate_batches()

    def _iterate_batches(self):
        if self.shuffle:
            document_ids = numpy.arange(self.batches_count)
            numpy.random.shuffle(document_ids)
        else:
            document_ids = numpy.arange(self.batches_count)

        for document_id in document_ids:
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