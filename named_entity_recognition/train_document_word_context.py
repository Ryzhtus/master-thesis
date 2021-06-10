from named_entity_recognition.metrics import FMeasureStorage, AccuracyStorage
from named_entity_recognition.utils import clear_tags, calculate_mean_context_vectors
from seqeval.metrics import performance_measure
from tqdm import tqdm

import torch
import torch.nn as nn

def train_epoch(model, criterion, optimizer, data, tag2idx, idx2tag, device, documents=None,
                scheduler=None, clip_grad=False, name=None):
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()
    epoch_repeated_entities_accuracy = AccuracyStorage()

    model.train()
    with tqdm(total=len(data)) as progress_bar:
        for batch in data:
            tokens = batch[0].to(device)
            tags = batch[1].to(device)
            masks = batch[2]
            document_ids = batch[3]
            sentences_ids = batch[4]

            if documents:
                for param in model.bert.parameters():
                    param.requires_grad = False

                documents_set = set(document_ids)

                mean_embeddings_for_batch_documents = {}
                sentences_from_documents = {}

                for document_id in documents_set:
                    mean_embeddings_for_batch_documents[document_id] = model.get_document_context(documents[document_id].to(device))
                    sentences_from_documents[document_id] = documents.get_document_words_by_sentences(document_id)

                for param in model.bert.parameters():
                    param.requires_grad = True

            batch_element_length = len(tags[0])

            predictions = model(tokens, document_ids, sentences_ids, mean_embeddings_for_batch_documents, sentences_from_documents)
            predictions = predictions.view(-1, predictions.shape[-1])

            tags_mask = tags != tag2idx['[PAD]']
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
            if clip_grad:
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


def eval_epoch(model, criterion, data, tag2idx, idx2tag, device, documents=None, name=None):
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
                document_ids = batch[3]
                sentences_ids = batch[4]

                if documents:
                    for param in model.bert.parameters():
                        param.requires_grad = False

                    documents_set = set(document_ids)

                    mean_embeddings_for_batch_documents = {}
                    sentences_from_documents = {}

                    for document_id in documents_set:
                        mean_embeddings_for_batch_documents[document_id] = model.get_document_context(
                            documents[document_id].to(device))
                        sentences_from_documents[document_id] = documents.get_document_words_by_sentences(document_id)

                    for param in model.bert.parameters():
                        param.requires_grad = True

                batch_element_length = len(tags[0])

                predictions = model(tokens, document_ids, sentences_ids, mean_embeddings_for_batch_documents,
                                    sentences_from_documents)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags_mask = tags != tag2idx['[PAD]']
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


def test_model(model, criterion, data, tag2idx, idx2tag, device, documents=None):
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
                document_ids = batch[3]
                sentences_ids = batch[4]

                if documents:
                    for param in model.bert.parameters():
                        param.requires_grad = False

                    documents_set = set(document_ids)

                    mean_embeddings_for_batch_documents = {}
                    sentences_from_documents = {}

                    for document_id in documents_set:
                        mean_embeddings_for_batch_documents[document_id] = model.get_document_context(
                            documents[document_id].to(device))
                        sentences_from_documents[document_id] = documents.get_document_words_by_sentences(document_id)

                    for param in model.bert.parameters():
                        param.requires_grad = True

                batch_element_length = len(tags[0])

                predictions = model(tokens, document_ids, sentences_ids, mean_embeddings_for_batch_documents,
                                    sentences_from_documents)
                predictions = predictions.view(-1, predictions.shape[-1])
                tags_mask = tags != tag2idx['[PAD]']
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


def train_model(model, criterion, optimizer, train_data, eval_data, train_documents, eval_documents, tag2idx, idx2tag,
                device, clip_grad, scheduler, epochs=1):
    for epoch in range(epochs):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs)
        train_epoch(model, criterion, optimizer, train_data, tag2idx, idx2tag, device, train_documents,
                    scheduler, clip_grad, name_prefix + 'Train:')
        eval_epoch(model, criterion, eval_data, tag2idx, idx2tag, device, eval_documents, name_prefix + 'Eval :')