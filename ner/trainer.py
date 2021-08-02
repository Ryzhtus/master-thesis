import comet_ml

from typing import List, Dict, Sequence
from ner.metrics import FMeasureStorage, AccuracyStorage
from ner.utils import clear_tags
from seqeval.metrics import performance_measure, classification_report, f1_score
from seqeval.scheme import IOB2
from tqdm import tqdm
from ner.document import Document

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim.optimizer


class Trainer():
    def __init__(self, experiment: comet_ml.Experiment, model, params: dict,
                 optimizer, criterion, scheduler, clip_grad: bool,
                 epochs: int, last_epoch: bool,
                 train_data: torch.utils.data.DataLoader,
                 eval_data: torch.utils.data.DataLoader,
                 test_data: torch.utils.data.DataLoader,
                 train_documents: Document,
                 eval_documents: Document,
                 test_documents: Document,
                 tag2idx: dict, idx2tag: dict,
                 device):
        self.model = model
        self.params = params

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_grad = clip_grad

        self.epochs = epochs
        self.last_epoch = last_epoch

        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

        self.train_documents = train_documents
        self.eval_documents = eval_documents
        self.test_documents = test_documents

        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

        self.device = device

        self.experiment = experiment
        self.experiment.log_parameters(self.params)

        self.train_loss = []
        self.eval_loss = []
        self.test_loss = []

        self.epoch_labels = []
        self.epoch_predictions = []

        self.progress_info = '{:>5s} Loss = {:.5f}, Token F1-score = {:.2%}, Span F1-score = {:.2%}'

    def __step(self, input_ids: torch.Tensor, tags: torch.Tensor, attention_masks: torch.Tensor, masks: List[List[int]],
               document_ids: List[int] = None, sentences_ids: List[int] = None,
               mean_embeddings_for_batch_documents: Dict = None, sentences_from_documents: Dict = None,
               freeze_bert: bool = False):

        if freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        if document_ids and sentences_ids and mean_embeddings_for_batch_documents and sentences_from_documents:
            predictions = self.model(input_ids, attention_masks, document_ids, sentences_ids, mean_embeddings_for_batch_documents,
                                     sentences_from_documents)
        else:
            predictions = self.model(input_ids, attention_masks)

        tags_mask = tags != -100
        tags_mask = tags_mask.view(-1)
        labels = torch.where(tags_mask, tags.view(-1), torch.tensor(self.criterion.ignore_index).type_as(tags))

        masks = masks.view(-1)

        loss = self.criterion(predictions.view(-1, predictions.shape[-1]), labels)

        predictions = predictions.argmax(dim=2).cpu().numpy()
        tags = tags.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_tags(tags, predictions, self.idx2tag)

        iteration_result = performance_measure(clear_labels, clear_predictions)

        self.epoch_labels += clear_labels
        self.epoch_predictions += clear_predictions

        return loss, iteration_result

    def __get_document_word_vectors(self, document_ids: List[int], documents: Document):
        for param in self.model.bert.parameters():
            param.requires_grad = False

        documents_set = set(document_ids)

        mean_embeddings_for_batch_documents = {}
        sentences_from_documents = {}

        for document_id in documents_set:
            mean_embeddings_for_batch_documents[document_id] = self.model.get_document_context(
                documents[document_id].to(self.device),
                documents.collect_all_positions_for_each_word(document_id))
            sentences_from_documents[document_id] = documents.get_document_words_by_sentences(
                document_id)

        for param in self.model.bert.parameters():
            param.requires_grad = True

        return mean_embeddings_for_batch_documents, sentences_from_documents

    def __train_epoch(self, name: str, freeze_bert=False):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()

        self.model.train()
        with self.experiment.train():
            with tqdm(total=len(self.train_data)) as progress_bar:
                for batch in self.train_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    masks = batch[2]
                    attention_masks = batch[3].to(self.device)

                    if self.train_documents:
                        document_ids = batch[5]
                        sentences_ids = batch[6]
                        mean_document_word_vectors, sentences_from_documents = self.__get_document_word_vectors(
                            document_ids, self.train_documents)
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, document_ids,
                                                    sentences_ids, mean_document_word_vectors,
                                                    sentences_from_documents, freeze_bert)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, freeze_bert=freeze_bert)

                    epoch_metrics + step_f1
                    epoch_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad:
                        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    torch.cuda.empty_cache()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, loss.item(), 0, 0))

                epoch_token_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions, scheme=IOB2)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data),
                                                                       epoch_token_f1_score, epoch_span_f1_score))

                self.experiment.log_metric("Train Token F1", epoch_token_f1_score)
                self.experiment.log_metric("Train Span F1", epoch_span_f1_score)
                self.experiment.log_metric("Train Recall", epoch_recall)
                self.experiment.log_metric("Train Precision", epoch_precision)

            self.train_loss.append(epoch_loss / len(self.train_data))

    def __eval_epoch(self, name):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()

        self.model.eval()
        with self.experiment.validate():
            with tqdm(total=len(self.eval_data)) as progress_bar:
                for batch in self.eval_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    masks = batch[2]
                    attention_masks = batch[3].to(self.device)

                    if self.eval_documents:
                        document_ids = batch[5]
                        sentences_ids = batch[6]
                        mean_document_word_vectors, sentences_from_documents = self.__get_document_word_vectors(
                            document_ids, self.eval_documents)
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, document_ids,
                                                    sentences_ids, mean_document_word_vectors,
                                                    sentences_from_documents)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks)

                    epoch_metrics + step_f1
                    epoch_loss += loss.item()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, loss.item(), 0, 0))

                epoch_token_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions, scheme=IOB2)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data),
                                                                       epoch_token_f1_score, epoch_span_f1_score))

                self.experiment.log_metric("Validation Token F1", epoch_token_f1_score)
                self.experiment.log_metric("Validation Span F1", epoch_span_f1_score)
                self.experiment.log_metric("Validation Recall", epoch_recall)
                self.experiment.log_metric("Validation Precision", epoch_precision)

            self.eval_loss.append(epoch_loss / len(self.eval_data))

    def __test_epoch(self, name: str):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()

        self.model.eval()
        with self.experiment.test():
            with tqdm(total=len(self.test_data)) as progress_bar:
                for batch in self.test_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    attention_masks = batch[3].to(self.device)
                    masks = batch[2]

                    if self.test_documents:
                        document_ids = batch[5]
                        sentences_ids = batch[6]
                        mean_document_word_vectors, sentences_from_documents = self.__get_document_word_vectors(
                            document_ids, self.test_documents)
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, document_ids,
                                                    sentences_ids, mean_document_word_vectors,
                                                    sentences_from_documents)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks)

                    epoch_metrics + step_f1
                    epoch_loss += loss.item()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, loss.item(), 0, 0))

                epoch_token_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                epoch_span_f1_score = f1_score(self.epoch_labels, self.epoch_predictions, scheme=IOB2)
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data),
                                                                       epoch_token_f1_score, epoch_span_f1_score))

                self.experiment.log_metric("Test Token F1", epoch_token_f1_score)
                self.experiment.log_metric("Test Span F1", epoch_span_f1_score)
                self.experiment.log_metric("Test Recall", epoch_recall)
                self.experiment.log_metric("Test Precision", epoch_precision)

            self.test_loss.append(epoch_loss / len(self.test_data))

    def fit(self):
        for epoch in range(self.epochs):
            progress = '[{} / {}] '.format(epoch + 1, self.epochs)

            if epoch == (self.epochs - 1):
                if self.last_epoch:
                    self.__train_epoch(progress + 'Train:', freeze_bert=True)
                else:
                    self.__train_epoch(progress + 'Train:')
            else:
                self.__train_epoch(progress + 'Train:')

            # clear labels and predictions from memory after training epoch
            self.epoch_labels = []
            self.epoch_predictions = []

            self.__eval_epoch(progress + 'Eval :')

            # clear labels and predictions from memory after validation epoch
            self.epoch_labels = []
            self.epoch_predictions = []

    def test(self):
        self.__test_epoch('Test :')

        print('Classification Report')
        print(classification_report(self.epoch_labels, self.epoch_predictions, scheme=IOB2, digits=4))