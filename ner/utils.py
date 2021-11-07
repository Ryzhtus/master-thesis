from typing import List, Dict

from ner.reader import ReaderCoNLL, ReaderOntonotes
from ner.dataset import CoNLLDatasetBERT, CoNLLDatasetT5, ChunksPlusDocumentsDataset
from ner.iterator import DocumentBatchIterator

from torch.utils.data import DataLoader


def create_dataset_and_standard_dataloader(model_name: str, dataset_name: str, filename: str, batch_size: int,
                                           shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL()
        sentences, tags, _ = reader.read(filename)
        if model_name == 'BERT':
            dataset = CoNLLDatasetBERT(sentences, tags, tokenizer)
        elif model_name == 'T5':
            dataset = CoNLLDatasetT5(sentences, tags, tokenizer)
        else:
            raise ValueError('The model {} name is not valid or this model is not supported.'.format(model_name))

        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes()
        sentences, tags, masks = reader.read(filename)
        dataset = CoNLLDatasetBERT(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def create_chunk_dataset_and_document_dataloader(dataset_name: str, filename: str, model_name: str, seq_length: int, batch_size: int,
                                                 shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL(include_document_ids=True)
        sentences, labels, _, document2sentences, sentence2position = reader.read(filename)
        dataset = ChunksPlusDocumentsDataset(sentences, labels, seq_length, document2sentences, sentence2position, tokenizer, model_name)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes(include_document_ids=True)
        sentences, labels, _, document2sentences, sentence2position = reader.read(filename)
        dataset = ChunksPlusDocumentsDataset(sentences, labels, seq_length, document2sentences, sentence2position, tokenizer, model_name)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def clear_for_metrics(labels, predictions, idx2tag, words_ids):
    """Актуальная функция для подготовки тэгов к подсчету метрик на entity-level"""
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
        for word_id in word_ids:
            clear_labels.append(non_pad_labels[word_id])
            clear_predictions.append(non_pad_predictions[word_id])

        y_true.append(clear_labels)
        y_pred.append(clear_predictions)

    return y_true, y_pred