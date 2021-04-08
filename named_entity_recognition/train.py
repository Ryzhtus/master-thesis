from named_entity_recognition.metrics import FMeasureStorage, AccuracyStorage
from seqeval.metrics import performance_measure
from tqdm import tqdm

import torch
import torch.nn as nn


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

def clear_tags_tensor(labels, predictions, masks, batch_element_length):
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
            sentence_labels.append(labels[idx])
            sentence_predictions.append(predictions[idx])
            if masks[idx] == 1:
                sentence_true_labels_mask.append(labels[idx])
                sentence_pred_labels_mask.append(predictions[idx])
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


def train_epoch(model, criterion, optimizer, data, tag2idx, idx2tag, device, scheduler=None, name=None):
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()
    epoch_repeated_entities_accuracy = AccuracyStorage()

    model.train()
    with tqdm(total=len(data)) as progress_bar:
        for batch in data:
            tokens = batch[0].to(device)
            tags = batch[1].to(device)
            masks = batch[2]

            batch_element_length = len(tags[0])

            predictions = model(tokens)
            predictions = predictions.view(-1, predictions.shape[-1])

            tags_mask = tags != tag2idx['<PAD>']
            tags_mask = tags_mask.view(-1)
            labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

            masks = masks.view(-1)

            loss = criterion(predictions, labels)

            predictions = predictions.argmax(dim=1)

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()

            # clear <PAD>, CLS and SEP tags from both labels and predictions
            clear_labels, clear_predictions, clear_repeated_entities_labels = clear_tags(labels, predictions, masks,
                                                                                         idx2tag, batch_element_length)

            iteration_result = performance_measure(clear_labels, clear_predictions)

            epoch_metrics + iteration_result
            epoch_repeated_entities_accuracy + clear_repeated_entities_labels
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if scheduler:
                scheduler.step()
            torch.cuda.empty_cache()

            progress_bar.update()
            progress_bar.set_description(
                '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'.format(name, loss.item(),
                                                                                                      0, 0))

        epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
        epoch_accuracy = epoch_repeated_entities_accuracy.report()
        progress_bar.set_description(
            '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'.format(name,
                                                                                                  epoch_loss / len(
                                                                                                      data),
                                                                                                  epoch_f1_score,
                                                                                                  epoch_accuracy))


def eval_epoch(model, criterion, data, tag2idx, idx2tag, device, name=None):
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()
    epoch_repeated_entities_accuracy = AccuracyStorage()

    model.eval()

    with torch.no_grad():
        with tqdm(total=len(data)) as progress_bar:
            for batch in data:
                tokens = batch[0].to(device)
                tags = batch[1].to(device)
                masks = batch[2]

                batch_element_length = len(tags[0])

                predictions = model(tokens)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags_mask = tags != tag2idx['<PAD>']
                tags_mask = tags_mask.view(-1)
                labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

                masks = masks.view(-1)

                loss = criterion(predictions, labels)

                predictions = predictions.argmax(dim=1)

                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                masks = masks.cpu().numpy()

                # clear <PAD>, CLS and SEP tags from both labels and predictions
                clear_labels, clear_predictions, clear_repeated_entities_labels = clear_tags(labels, predictions, masks,
                                                                                             idx2tag,
                                                                                             batch_element_length)

                iteration_result = performance_measure(clear_labels, clear_predictions)

                epoch_metrics + iteration_result
                epoch_repeated_entities_accuracy + clear_repeated_entities_labels
                epoch_loss += loss.item()

                progress_bar.update()
                progress_bar.set_description(
                    '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'.format(name,
                                                                                                          loss.item(),
                                                                                                          0, 0))

            epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
            epoch_accuracy = epoch_repeated_entities_accuracy.report()
            progress_bar.set_description(
                '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'.format(name,
                                                                                                      epoch_loss / len(
                                                                                                          data),
                                                                                                      epoch_f1_score,
                                                                                                      epoch_accuracy))

def test_model(model, criterion, data, tag2idx, idx2tag, device):
    name = '[Final] Test :'
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()
    epoch_repeated_entities_accuracy = AccuracyStorage()

    model.eval()

    with torch.no_grad():
        with tqdm(total=len(data)) as progress_bar:
            for batch in data:
                tokens = batch[0].to(device)
                tags = batch[1].to(device)
                masks = batch[2]

                batch_element_length = len(tags[0])

                predictions = model(tokens)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags_mask = tags != tag2idx['<PAD>']
                tags_mask = tags_mask.view(-1)
                labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

                masks = masks.view(-1)

                loss = criterion(predictions, labels)

                predictions = predictions.argmax(dim=1)

                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                masks = masks.cpu().numpy()

                # clear <PAD>, CLS and SEP tags from both labels and predictions
                clear_labels, clear_predictions, clear_repeated_entities_labels = clear_tags(labels, predictions, masks,
                                                                                             idx2tag,
                                                                                             batch_element_length)

                iteration_result = performance_measure(clear_labels, clear_predictions)

                epoch_metrics + iteration_result
                epoch_repeated_entities_accuracy + clear_repeated_entities_labels
                epoch_loss += loss.item()

                progress_bar.update()
                progress_bar.set_description(
                    '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'.format(name,
                                                                                                          loss.item(),
                                                                                                          0, 0))

            epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
            epoch_accuracy = epoch_repeated_entities_accuracy.report()
            progress_bar.set_description(
                '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'.format(name,
                                                                                                      epoch_loss / len(
                                                                                                          data),
                                                                                                      epoch_f1_score,
                                                                                                      epoch_accuracy))


def train_model(model, criterion, optimizer, train_data, eval_data, tag2idx, idx2tag, device, scheduler, epochs=1):
    for epoch in range(epochs):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs)
        train_epoch(model, criterion, optimizer, train_data, tag2idx, idx2tag, device, scheduler,
                    name_prefix + 'Train:')
        eval_epoch(model, criterion, eval_data, tag2idx, idx2tag, device, name_prefix + 'Eval :')


