from named_entity_recognition.reader import ReaderCoNLL, ReaderOntonotes, ReaderDocumentCoNLL, ReaderDocumentOntonotes
from named_entity_recognition.dataset import CoNLLDataset, SentencesDataset, SentencesPlusDocumentsDataset
from named_entity_recognition.iterator import DocumentBatchIterator
from named_entity_recognition.document import Document

from torch.utils.data import DataLoader


def create_dataset_and_standard_dataloader(dataset_name: str, filename: str, batch_size: int, shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderCoNLL()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = CoNLLDataset(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderOntonotes()
        sentences, tags, masks = reader.get_sentences(filename)
        dataset = CoNLLDataset(sentences, tags, masks, tokenizer)
        return dataset, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def create_dataset_and_document_dataloader(dataset_name: str, filename: str, batch_size: int, shuffle: bool, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderDocumentCoNLL()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = SentencesPlusDocumentsDataset(sentences, tags, masks, document2sentences, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        return dataset, documents, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)

    if dataset_name == 'ontonotes':
        reader = ReaderDocumentOntonotes()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = SentencesPlusDocumentsDataset(sentences, tags, masks, document2sentences, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        return dataset, documents, DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.paddings)


def create_dataset_and_document_level_iterator(dataset_name: str, filename: str, group_documents: bool,
                                               batch_size: int, tokenizer):
    if dataset_name == 'conll':
        reader = ReaderDocumentCoNLL()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = SentencesDataset(sentences, tags, masks, tokenizer)
        documents = Document(sentences, document2sentences, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences, group_documents=group_documents,
                                              batch_size=batch_size, shuffle=True)

        return dataset, documents, data_iterator

    if dataset_name == 'ontonotes':
        reader = ReaderDocumentOntonotes()
        sentences, tags, masks, document2sentences = reader.get_sentences(filename)
        dataset = SentencesDataset(sentences, tags, masks, tokenizer)
        data_iterator = DocumentBatchIterator(dataset, document2sentences, shuffle=True)

        return dataset, data_iterator


def clear_tags(labels, predictions, masks, idx2tag, batch_element_length):
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