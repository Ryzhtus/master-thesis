import torch 
import torch.nn as nn

from pytorch_lightning import LightningModule

from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup

from ner.utils import clear_for_metrics
from seqeval import f1_score

class LightningBERT(LightningModule):
    def __init__(self, model, params, idx2tag, train, valid, test):
        super().__init__()
        # model
        self.model = model

        # make variables for storing true and pred labels from each batch
        self.train_epoch_labels = []
        self.train_epoch_predictions = []
        
        self.val_epoch_labels = []
        self.val_epoch_predictions = []
        
        self.test_epoch_labels = []
        self.test_epoch_predictions = []

        self.train_dataset = train
        self.valid_dataset = valid
        self.test_dataset = test

        self.params = params
        self.idx2tag = idx2tag

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params['weight_decay'],
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.params['optimizer'] == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),  # according to RoBERTa paper
                              lr=self.params['learning_rate'],
                              eps=self.params['adam_eps'])
        
        elif self.params['optimizer'] == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.params['learning_rate'],
                                          eps=self.params['adam_eps'],
                                          weight_decay=self.params['weight_decay'])
        else:
            raise ValueError("Optimizer type does not exist.")
        num_gpus = len([x for x in str(self.params['gpus']).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.params['accumulate_grad_batches'] * num_gpus) + 1) * self.params['max_epochs']
        warmup_steps = int(self.params['warmup_proportion'] * t_total)

        if self.params['lr_scheduler'] == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.params['learning_rate'], pct_start=float(warmup_steps/t_total),
                final_div_factor=self.params['final_div_factor'],
                total_steps=t_total, anneal_strategy='linear')
        elif self.params['lr_scheduler'] == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif self.params['lr_scheduler'] == "polydecay":
            if self.params['learning_rate_mini'] == -1:
                lr_mini = self.params['learning_rate'] / self.params['polydecay_ratio']
            else:
                lr_mini = self.params['learning_rate_mini']
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total, lr_end=lr_mini)
        else:
            raise ValueError

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def __step(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor,
               words_ids: list[list[int]], mode: str):
        """
        Общий метод, используемый для шага обучения, валидации и тестирования.
        В этом блоке модель делает предсказание по батчу, считается значение функции потерь и полученные предсказания
        сохраняются в списке, чтобы в конце эпохе честно посчитать метрики по сущносят
        """

        predictions = self.model(input_ids=input_ids, attention_mask=attention_mask)

        loss = self.criterion(predictions.view(-1, predictions.shape[-1]), labels.view(-1))

        predictions = predictions.argmax(dim=2).cpu().numpy()
        labels = labels.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_for_metrics(labels, predictions, self.idx2tag, words_ids)

        if mode == 'train':
            self.train_epoch_labels += clear_labels
            self.train_epoch_predictions += clear_predictions
        elif mode == 'val':
            self.val_epoch_labels += clear_labels
            self.val_epoch_predictions += clear_predictions
        elif mode == 'test':
            self.test_epoch_labels += clear_labels
            self.test_epoch_predictions += clear_predictions

        return loss

    def training_step(self, batch, _):
        input_ids, labels, attention_mask, words_ids = batch

        loss = self.__step(input_ids, labels, attention_mask, words_ids, 'train')
        metric = f1_score(self.train_epoch_labels, self.train_epoch_predictions)

        self.log('Train Step F1', metric, prog_bar=True)

        return loss

    def training_epoch_end(self, _):
        epoch_metric = f1_score(self.train_epoch_labels, self.train_epoch_predictions)

        self.log('Train Epoch F1', epoch_metric, prog_bar=True)

        self.train_epoch_labels = []
        self.train_epoch_predictions = []

    def validation_step(self, batch, _):
        input_ids, labels, attention_mask, words_ids = batch

        loss = self.__step(input_ids, labels, attention_mask, words_ids, 'val')
        metric = f1_score(self.val_epoch_labels, self.val_epoch_predictions)

        self.log('Val Step F1', metric, prog_bar=True)
        self.log("Val Loss", loss, prog_bar=True)

        return loss

    def validation_epoch_end(self, _):
        epoch_metric = f1_score(self.val_epoch_labels, self.val_epoch_predictions)

        self.log('Val Epoch F1', epoch_metric, prog_bar=True)

        self.val_epoch_labels = []
        self.val_epoch_predictions = []

    def test_step(self, batch, _):
        input_ids, labels, attention_mask, words_ids = batch

        loss = self.__step(input_ids, labels, attention_mask, words_ids, 'test')
        metric = f1_score(self.test_epoch_labels, self.test_epoch_predictions)

        self.log('Test Step F1', metric, prog_bar=True)
        self.log("Test Loss", loss, prog_bar=True)

        return loss

    def test_epoch_end(self, _):
        epoch_metric = f1_score(self.test_epoch_labels, self.test_epoch_predictions)

        self.log('Test Epoch F1', epoch_metric, prog_bar=True)

        self.test_epoch_labels = []
        self.test_epoch_predictions = []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, self.params['batch_size'], shuffle=True, collate_fn=self.train_dataset.paddings)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, self.params['batch_size'], shuffle=False, collate_fn=self.valid_dataset.paddings)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, self.params['batch_size'], shuffle=False, collate_fn=self.test_dataset.paddings)