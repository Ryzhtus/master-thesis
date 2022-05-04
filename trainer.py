import torch 
import torch.nn as nn 

from utils import transform_predictions_to_labels

from pytorch_lightning import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report

class LightningBert(LightningModule):
    def __init__(self, model, params, idx2tag, train, valid, test, rdf=None, scheme=None):
        super().__init__()
        # model
        self.model = model
        self.num_labels = 73
        self.scheme = scheme
        self.rdf = rdf

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

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params['weight_decay'],
            },
            {
                "params": [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.classification_head.bilstm.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params['weight_decay'],
                "lr": self.params["lstm_lr"]
            },
            {
                "params": [p for n, p in self.model.classification_head.bilstm.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.classification_head.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params['weight_decay'],
                "lr": self.params["lstm_lr"]
            },
            {
                "params": [p for n, p in self.model.classification_head.classifier.named_parameters() if any(nd in n for nd in no_decay)],
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
    
    def compute_loss(self, sequence_logits, sequence_labels, input_mask=None):
        if input_mask is not None:
            active_loss = input_mask.view(-1) == 1
            active_logits = sequence_logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, sequence_labels.view(-1), torch.tensor(self.criterion.ignore_index).type_as(sequence_labels)
            )
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(sequence_logits.view(-1, self.num_labels), sequence_labels.view(-1))
        return loss

    def __step(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor, wordpiece_mask, mode: str, features=None):
        """
        Общий метод, используемый для шага обучения, валидации и тестирования.
        В этом блоке модель делает предсказание по батчу, считается значение функции потерь и полученные предсказания
        сохраняются в списке, чтобы в конце эпохе честно посчитать метрики по сущносят
        """
        if features != None:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, features=features)
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = self.compute_loss(logits, labels, input_mask=attention_mask)

        clear_predictions = transform_predictions_to_labels(logits, wordpiece_mask, self.idx2tag, input_type="logit")
        clear_labels = transform_predictions_to_labels(labels, wordpiece_mask, self.idx2tag, input_type="label")
        
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
        if self.rdf:
            input_ids, labels, attention_mask, words_ids, is_wordpiece_mask, sentence_chunks = batch
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_masks"]
            wordpiece_mask = batch["wordpiece_masks"]
        
        if self.params["add_features"]:
            features = batch["features"]
            loss = self.__step(input_ids, labels, attention_mask, wordpiece_mask, 'train', features=features)
        else:
            loss = self.__step(input_ids, labels, attention_mask, wordpiece_mask, 'train')

        self.log('train_loss', loss, prog_bar=True)

        return loss

    def training_epoch_end(self, _):
        if self.scheme:
            epoch_metric = f1_score(self.train_epoch_labels, self.train_epoch_predictions, mode='strict', scheme=self.scheme)
        else:
            epoch_metric = f1_score(self.train_epoch_labels, self.train_epoch_predictions)

        self.log('train_f1', epoch_metric, prog_bar=True)

        self.train_epoch_labels = []
        self.train_epoch_predictions = []

    def validation_step(self, batch, _):
        if self.rdf:
            input_ids, labels, attention_mask, words_ids, is_wordpiece_mask, sentence_chunks = batch
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_masks"]
            wordpiece_mask = batch["wordpiece_masks"]

        if self.params["add_features"]:
            features = batch["features"]
            loss = self.__step(input_ids, labels, attention_mask, wordpiece_mask, 'val', features=features)
        else:
            loss = self.__step(input_ids, labels, attention_mask, wordpiece_mask, 'val')

        self.log("val_loss", loss, prog_bar=True)

        return loss

    def validation_epoch_end(self, _):
        if self.scheme:
            epoch_metric = f1_score(self.val_epoch_labels, self.val_epoch_predictions, mode='strict', scheme=self.scheme)
        else:
            epoch_metric = f1_score(self.val_epoch_labels, self.val_epoch_predictions)

        self.log('val_f1', epoch_metric, prog_bar=True)

        self.val_epoch_labels = []
        self.val_epoch_predictions = []

    def test_step(self, batch, _):
        if self.rdf:
            input_ids, labels, attention_mask, words_ids, is_wordpiece_mask, sentence_chunks = batch
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_masks"]
            wordpiece_mask = batch["wordpiece_masks"]

        if self.params["add_features"]:
            features = batch["features"]
            loss = self.__step(input_ids, labels, attention_mask, wordpiece_mask, 'test', features=features)
        else:
            loss = self.__step(input_ids, labels, attention_mask, wordpiece_mask, 'test')

        self.log("test_loss", loss, prog_bar=True)

        return loss

    def test_epoch_end(self, _):
        if self.scheme:
            epoch_metric = f1_score(self.test_epoch_labels, self.test_epoch_predictions, mode='strict', scheme=self.scheme)
        else:   
            epoch_metric = f1_score(self.test_epoch_labels, self.test_epoch_predictions)

        self.log('test_f1', epoch_metric, prog_bar=True)

        if self.scheme:
            print(classification_report(self.test_epoch_labels, self.test_epoch_predictions, digits=4, mode='strict', scheme=self.scheme))
        else:
            print(classification_report(self.test_epoch_labels, self.test_epoch_predictions, digits=4))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           self.params['batch_size'], 
                                           shuffle=self.params['shuffle_train_eval'], 
                                           pin_memory=True,
                                           collate_fn=self.train_dataset.paddings)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_dataset, 
                                           self.params['batch_size'], 
                                           shuffle=False, 
                                           pin_memory=True,
                                           collate_fn=self.valid_dataset.paddings)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, 
                                           self.params['batch_size'], 
                                           shuffle=False, 
                                           collate_fn=self.test_dataset.paddings)