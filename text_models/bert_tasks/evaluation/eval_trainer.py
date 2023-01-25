import logging
from typing import List, Optional, Dict

from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, BertModel, PreTrainedTokenizerBase

from text_models.bert_tasks.evaluation.save_loss import write_csv_loss

logger = logging.getLogger(__name__)

class ValMetric:
    LOSS_DOCS = "loss"
    LOSS_TASK = "loss_task"
    TASK = "task_map"


class IREvalTrainer(Trainer):
    class EvalTransformer(nn.Module):
        def __init__(self, model, tokenizer, max_len):
            super(IREvalTrainer.EvalTransformer, self).__init__()
            self.model = model
            self.tokenizer = tokenizer
            self.max_len = max_len

        def forward(self, features):
            trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
            if 'token_type_ids' in features:
                trans_features['token_type_ids'] = features['token_type_ids']

            output_states = self.model(**trans_features, return_dict=False)
            output_tokens = output_states[0]

            features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})
            return features

        def get_word_embedding_dimension(self) -> int:
            return self.model.config.hidden_size

        def tokenize(self, texts: List[str]):
            output = {}
            to_tokenize = [[str(s).strip() for s in col] for col in [texts]]
            output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt",
                                         max_length=self.max_len))
            return output

    def set_env_vars(
        self,
        evaluator: SentenceEvaluator,
        model: BertModel,
        tokenizer: PreTrainedTokenizerBase,
        val_task_dataset: Dataset,
        max_len: int = 512,
        device: str = "cpu",
    ):
        self.evaluator = evaluator
        self.val_task_dataset = val_task_dataset

        transformer = IREvalTrainer.EvalTransformer(model, tokenizer, max_len)
        pooling = models.Pooling(transformer.get_word_embedding_dimension())
        self.eval_model = SentenceTransformer(modules=[transformer, pooling], device=device)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = {}
        epoch = self.state.epoch if self.state.epoch is not None else -1

        if eval_dataset is not None or self.eval_dataset is not None:
            metrics = super().evaluate(self.eval_dataset, ignore_keys, metric_key_prefix)
            loss_task_val = super().evaluate(self.val_task_dataset, ignore_keys, metric_key_prefix)[f"{metric_key_prefix}_loss"]
            metrics[f"{metric_key_prefix}_{ValMetric.LOSS_TASK}"] = loss_task_val
            if self.args.output_dir is not None:
                write_csv_loss(metrics[f"{metric_key_prefix}_loss"], loss_task_val, self.args.output_dir, epoch, self.state.global_step)

        map_value = self.evaluator(
            self.eval_model,
            output_path=self.args.output_dir,
            epoch=epoch,
            steps=self.state.global_step,
        )
        metrics[f"{metric_key_prefix}_{ValMetric.TASK}"] = map_value
        return metrics
