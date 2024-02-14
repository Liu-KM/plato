"""Entities needed to conduct federated learning with LoRA adapters."""

import logging
import math
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from torch.optim import SGD
import evaluate
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from plato.algorithms import fedavg
from plato.datasources import base
from plato.trainers import huggingface,basic
from plato.config import Config
from plato.models import registry as model_registry
from peft import (
    get_peft_model,
    LoraConfig,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)


class LoraModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        base_model = AutoModelForSequenceClassification.from_pretrained(Config().trainer.model_name, return_dict=True)
        lora_config = Config().parameters.lora
        self.base_model = get_peft_model(
            base_model, LoraConfig(**lora_config._asdict())
        )
        self.base_model.print_trainable_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class Trainer(huggingface.Trainer):
    """A trainer with custom training and testing loops for LoRA fine-tuning."""

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The training loop for HuggingFace models.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        sampler: the sampler that extracts a partition for this client.
        """
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "right"

        self.training_args.num_train_epochs = config["epochs"]
        self.training_args.per_device_train_batch_size = config["batch_size"]

        self.trainer = huggingface.SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=trainset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(
                self.tokenizer,
            ),
            sampler=sampler,
            callbacks=self.trainer_callbacks,
        )

        self.trainer.train()

    def test_model(
        self, config, testset, sampler=None, **kwargs
    ):  # pylint: disable=unused-argument
        """The testing loop for HuggingFace models.

        Arguments:
            config: Configuration parameters as a dictionary.
            testset: The test dataset.
        """
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "right"

        self.training_args.per_device_eval_batch_size = config["batch_size"]
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # 将 logits 转换为二分类预测结果（0或1），基于 logit 的最大值
            predictions = np.argmax(logits, axis=-1)
            # 计算准确率
            acc = accuracy_score(labels, predictions)
            # 计算精确率、召回率和 F1 分数，适用于二分类任务
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }
        self.trainer = huggingface.SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=testset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(
                self.tokenizer,
            ),
            sampler=sampler,
            callbacks=None,
            compute_metrics=compute_metrics,
        )

        metrics = self.trainer.evaluate()

        # try:
        perplexity = math.exp(metrics["eval_loss"])
        # except OverflowError:
        #     perplexity = float("inf")
        logging.info(f"The metrics is {metrics}")
        return metrics["eval_accuracy"]

    def save_model(self):
        logging.info("Skip trainer saving.")


class DataSource(base.DataSource):
    """A datasource with custom training and validation datasets for LoRA fine-tuning."""

    def __init__(self):
        super().__init__()

        dataset_name = Config().data.dataset_name
        task = Config().data.task
        logging.info("Dataset: %s(%s)", dataset_name,task)

        dataset = load_dataset(dataset_name,task)

        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model_name = Config().trainer.model_name
        if "llama" in model_name:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.padding_side = "right"
        if getattr(tokenizer, "pad_token_id") is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        def tokenize_function(examples):
            # max_length=None => use the model max length (it's actually the default)
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
            return outputs

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        train_data = tokenized_datasets["train"].shuffle(seed=42)
        val_data = tokenized_datasets["validation"].shuffle(seed=42)

        self.trainset = train_data
        self.testset = val_data


class Algorithm(fedavg.Algorithm):
    def extract_weights(self, model=None):
        # Extract LoRA wegiths
        return {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(self.model.base_model).items()
        }

    def load_weights(self, weights):
        # Load LoRA weights
        return set_peft_model_state_dict(self.model.base_model, weights)

