from typing import List, Dict

from ner.reader import ReaderCoNLL, ReaderOntonotes
from ner.dataset import CoNLLDatasetBERT, CoNLLDatasetT5, SentencesDataset, SentencesPlusDocumentsDataset
from ner.iterator import DocumentBatchIterator
from ner.document import Document

from torch.utils.data import DataLoader


def create_dataset_and_standard_dataloader(model_name: str, dataset_name: str, filename: str, batch_size: int,
                                           shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL()
        sentences, tags, masks = reader.read(filename)
        if model_name == 'BERT':
            dataset = CoNLLDatasetBERT(sentences, tags, masks, tokenizer)
        elif model_name == 'T5':
            dataset = CoNLLDatasetT5(sentences, tags, masks, tokenizer)
        else:
            raise ValueError('The model {} name is not valid or this model is not supported.'.format(model_name))

        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes()
        sentences, tags, masks = reader.read(filename)
        dataset = CoNLLDatasetBERT(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def create_dataset_and_document_dataloader(dataset_name: str, filename: str, batch_size: int, shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL(include_document_ids=True)
        sentences, labels, _, document2sentences, sentence2position = reader.read(filename)
        dataset = SentencesPlusDocumentsDataset(sentences, labels, document2sentences, sentence2position, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        return dataset, documents, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes(include_document_ids=True)
        sentences, labels, _, document2sentences, sentence2position = reader.read(filename)
        dataset = SentencesPlusDocumentsDataset(sentences, labels, document2sentences, sentence2position, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        return dataset, documents, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def create_dataset_and_document_level_iterator(dataset_name: str, filename: str, group_documents: bool,
                                               batch_size: int, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL(include_document_ids=True)
        sentences, tags, masks, document2sentences = reader.read(filename)
        dataset = SentencesDataset(sentences, tags, masks, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences, group_documents=group_documents,
                                              batch_size=batch_size, shuffle=True)

        return dataset, documents, data_iterator

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes(include_document_ids=True)
        sentences, tags, masks, document2sentences = reader.read(filename)
        dataset = SentencesDataset(sentences, tags, masks, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences, shuffle=True)

        return dataset, data_iterator

def clear_for_metrics(labels: List[List[int]], predictions: List[List[int]], idx2tag: Dict, words_ids: List[List[int]]):
    y_true = []
    y_pred = []

    for label_list, preds_list, word_ids in zip(labels, predictions, words_ids):
        non_pad_labels = []
        non_pad_predictions = []

        clear_labels = []
        clear_predictions = []

        # убираем PAD, CLS и SEP токены
        for idx in range(len(list(label_list))):
            if label_list[idx] != -100:
                non_pad_labels.append(idx2tag[label_list[idx]])
                non_pad_predictions.append(idx2tag[preds_list[idx]])

        # собираем только тэги, проставленые первому сабтокену слова
        # добавляем первый тэг по умолчанию
        clear_labels.append(non_pad_labels[0])
        clear_predictions.append(non_pad_predictions[0])
        # если предыдущий индекс слова = текущему, то это тоже слово
        for subtoken_id in range(1, len(word_ids)):
            if word_ids[subtoken_id] != word_ids[subtoken_id - 1]:
                clear_labels.append(non_pad_labels[subtoken_id])
                clear_predictions.append(non_pad_predictions[subtoken_id])

        y_true.append(clear_labels)
        y_pred.append(clear_predictions)

    return y_true, y_pred

def clear_tags(labels, predictions, idx2tag):
    y_true = []
    y_pred = []

    for label_list, preds_list in zip(labels, predictions):
        true_tags = label_list != -100

        clear_labels = []
        clear_predictions = []
        for idx in range(len(true_tags)):
            if true_tags[idx] == True:
                clear_labels.append(idx2tag[label_list[idx]])
                clear_predictions.append(idx2tag[preds_list[idx]])

        y_true.append(clear_labels)
        y_pred.append(clear_predictions)

    return y_true, y_pred

def clear_tags_old(labels, predictions, masks, idx2tag, batch_element_length):
    """ this function removes <PAD>, CLS and SEP tags at each sentence
        and convert both ids of tags and batch elements to SeqEval input format
        [[first sentence tags], [second sentence tags], ..., [last sentence tags]]"""

    clear_labels = []
    clear_predictions = []
    masked_true_labels = []
    masked_pred_labels = []

    sentence_labels = []
    sentence_predictions = []
    sentence_true_labels_mask = []
    sentence_pred_labels_mask = []

    sentence_length = 0

    for idx in range(len(labels)):
        if labels[idx] != 0:
            sentence_labels.append(idx2tag[labels[idx]])
            sentence_predictions.append(idx2tag[predictions[idx]])
            if masks[idx] == 1:
                sentence_true_labels_mask.append(idx2tag[labels[idx]])
                sentence_pred_labels_mask.append(idx2tag[predictions[idx]])
            sentence_length += 1

            if sentence_length == batch_element_length:
                # not including the 0 and the last element of list, because of CLS and SEP tokens
                clear_labels.append(sentence_labels[1: len(sentence_labels) - 1])
                clear_predictions.append(sentence_predictions[1: len(sentence_predictions) - 1])
                masked_true_labels.append(sentence_true_labels_mask[1: len(sentence_true_labels_mask) - 1])
                masked_pred_labels.append(sentence_pred_labels_mask[1: len(sentence_pred_labels_mask) - 1])
                sentence_labels = []
                sentence_predictions = []
                sentence_true_labels_mask = []
                sentence_pred_labels_mask = []
                sentence_length = 0
        else:
            if sentence_labels:
                clear_labels.append(sentence_labels[1: len(sentence_labels) - 1])
                clear_predictions.append(sentence_predictions[1: len(sentence_predictions) - 1])
                masked_true_labels.append(sentence_true_labels_mask[1: len(sentence_true_labels_mask) - 1])
                masked_pred_labels.append(sentence_pred_labels_mask[1: len(sentence_pred_labels_mask) - 1])
                sentence_labels = []
                sentence_predictions = []
                sentence_true_labels_mask = []
                sentence_pred_labels_mask = []
            else:
                pass

    masked_true_labels = [element for element in masked_true_labels if element != []]
    masked_pred_labels = [element for element in masked_pred_labels if element != []]
    repeated_entities_labels = {'true': masked_true_labels, 'pred': masked_pred_labels}

    return clear_labels, clear_predictions, repeated_entities_labels