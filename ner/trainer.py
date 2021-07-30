import comet_ml

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

        self.test_labels = []
        self.test_predictions = []

        self.progress_info = '{:>5s} Loss = {:.5f}, F1-score = {:.2%}, Repeated Entities Accuracy = {:.2%}'

    def __step(self, tokens, tags, attention_masks, masks, test_epoch, document_ids=None, sentences_ids=None,
               mean_embeddings_for_batch_documents=None, sentences_from_documents=None, freeze_bert=False):

        batch_element_length = len(tags[0])

        if freeze_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        if document_ids and sentences_ids and mean_embeddings_for_batch_documents and sentences_from_documents:
            predictions = self.model(tokens, document_ids, sentences_ids, mean_embeddings_for_batch_documents,
                                     sentences_from_documents)
        else:
            predictions = self.model(tokens, attention_masks)

        predictions_for_metrics = predictions.argmax(dim=2).cpu().numpy()

        predictions = predictions.view(-1, predictions.shape[-1])

        tags_mask = tags != -100
        tags_mask = tags_mask.view(-1)
        labels = torch.where(tags_mask, tags.view(-1), torch.tensor(self.criterion.ignore_index).type_as(tags))

        masks = masks.view(-1)

        loss = self.criterion(predictions, labels)

        predictions = predictions.argmax(dim=1)

        predictions = predictions.cpu().numpy()
        tags = tags.cpu().numpy()
        labels = labels.cpu().numpy()
        masks = masks.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_tags(tags, predictions_for_metrics, self.idx2tag)

        iteration_result = performance_measure(clear_labels, clear_predictions)

        if test_epoch:
            self.test_labels += clear_labels
            self.test_predictions += clear_predictions

        return loss, iteration_result

    def __get_document_word_vectors(self, document_ids: list, documents: Document):
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

    def __train_epoch(self, name, freeze_bert=False):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()
        epoch_repeated_entities_accuracy = AccuracyStorage()

        self.model.train()
        with self.experiment.train():
            with tqdm(total=len(self.train_data)) as progress_bar:
                for batch in self.train_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    masks = batch[2]
                    attention_masks = batch[3].to(self.device)
                    words_ids = batch[4]

                    if self.train_documents:
                        document_ids = batch[5]
                        sentences_ids = batch[6]
                        mean_document_word_vectors, sentences_from_documents = self.__get_document_word_vectors(
                            document_ids, self.train_documents)
                        loss, step_f1 = self.__step(tokens, tags, masks, False, document_ids,
                                                    sentences_ids, mean_document_word_vectors,
                                                    sentences_from_documents, freeze_bert)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, False, freeze_bert=freeze_bert)

                    epoch_metrics + step_f1
                    # epoch_repeated_entities_accuracy + step_re_accuracy
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

                epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                # epoch_accuracy = epoch_repeated_entities_accuracy.report()
                epoch_accuracy = 0
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.train_data),
                                                                       epoch_f1_score, epoch_accuracy))

                self.experiment.log_metric("Train F1", epoch_f1_score)
                self.experiment.log_metric("Train Recall", epoch_recall)
                self.experiment.log_metric("Train Precision", epoch_precision)
                self.experiment.log_metric("Train RE Accuracy", epoch_accuracy)

            self.train_loss.append(epoch_loss / len(self.train_data))

    def __eval_epoch(self, name):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()
        epoch_repeated_entities_accuracy = AccuracyStorage()

        self.model.eval()
        with self.experiment.validate():
            with tqdm(total=len(self.eval_data)) as progress_bar:
                for batch in self.eval_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    masks = batch[2]
                    attention_masks = batch[3].to(self.device)
                    words_ids = batch[4]

                    if self.eval_documents:
                        document_ids = batch[5]
                        sentences_ids = batch[6]
                        mean_document_word_vectors, sentences_from_documents = self.__get_document_word_vectors(
                            document_ids, self.eval_documents)
                        loss, step_f1 = self.__step(tokens, tags, masks, False, document_ids,
                                                    sentences_ids, mean_document_word_vectors,
                                                    sentences_from_documents)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, False)

                    epoch_metrics + step_f1
                    # epoch_repeated_entities_accuracy + step_re_accuracy
                    epoch_loss += loss.item()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, loss.item(), 0, 0))

                epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                # epoch_accuracy = epoch_repeated_entities_accuracy.report()
                epoch_accuracy = 0
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.eval_data),
                                                                       epoch_f1_score, epoch_accuracy))

                self.experiment.log_metric("Validation F1", epoch_f1_score)
                self.experiment.log_metric("Validation Recall", epoch_recall)
                self.experiment.log_metric("Validation Precision", epoch_precision)
                self.experiment.log_metric("Validation Accuracy", epoch_accuracy)

            self.eval_loss.append(epoch_loss / len(self.eval_data))

    def __test_epoch(self, name):
        epoch_loss = 0
        epoch_metrics = FMeasureStorage()
        epoch_repeated_entities_accuracy = AccuracyStorage()

        self.model.eval()
        with self.experiment.test():
            with tqdm(total=len(self.test_data)) as progress_bar:
                for batch in self.test_data:
                    tokens = batch[0].to(self.device)
                    tags = batch[1].to(self.device)
                    masks = batch[2]
                    attention_masks = batch[3].to(self.device)
                    words_ids = batch[4]

                    if self.test_documents:
                        document_ids = batch[5]
                        sentences_ids = batch[6]
                        mean_document_word_vectors, sentences_from_documents = self.__get_document_word_vectors(
                            document_ids, self.test_documents)
                        loss, step_f1 = self.__step(tokens, tags, masks, True, document_ids,
                                                    sentences_ids, mean_document_word_vectors,
                                                    sentences_from_documents)
                    else:
                        loss, step_f1 = self.__step(tokens, tags, attention_masks, masks, True)

                    epoch_metrics + step_f1
                    # epoch_repeated_entities_accuracy + step_re_accuracy
                    epoch_loss += loss.item()

                    progress_bar.update()
                    progress_bar.set_description(self.progress_info.format(name, loss.item(), 0, 0))

                epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
                # epoch_accuracy = epoch_repeated_entities_accuracy.report()
                epoch_accuracy = 0
                progress_bar.set_description(self.progress_info.format(name, epoch_loss / len(self.test_data),
                                                                       epoch_f1_score, epoch_accuracy))

                self.experiment.log_metric("Test F1", epoch_f1_score)
                self.experiment.log_metric("Test Recall", epoch_recall)
                self.experiment.log_metric("Test Precision", epoch_precision)
                self.experiment.log_metric("Test RE Accuracy", epoch_accuracy)

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

            self.__eval_epoch(progress + 'Eval :')

    def test(self):
        self.__test_epoch('Test :')

        print('Brute F1-score: {}'.format(f1_score(self.test_labels, self.test_predictions, mode='strict', scheme=IOB2)))

        print('Classification Report')
        print(classification_report(self.test_labels, self.test_predictions, mode='strict', scheme=IOB2))