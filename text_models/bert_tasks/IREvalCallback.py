from typing import List, Union

import numpy as np
import torch
from sentence_transformers import evaluation
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from text_models.datasets import BertModelDataset


class IREvalCallback(TrainerCallback):
    def __init__(self, evaluator: evaluation.InformationRetrievalEvaluator, model, tokenizer, task, max_len, device):
        self.evaluator = evaluator
        self.eval_model = IREvalCallback.EvalModel(model, tokenizer, task, max_len, device)

    class EvalModel:
        def __init__(self, model, tokenizer, task, max_len, device):
            self.model = model
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.device = torch.device(device)
            self.task = task

        def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = "sentence_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
        ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
            if device is not None:
                self.device = torch.device(device)

            encoded_input = self.tokenizer(
                sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
            )
            dataset = BertModelDataset(encoded_input)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            result = []

            loop = loader
            if show_progress_bar:
                loop = tqdm(loader, leave=True)
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                with torch.no_grad():
                    model_output = self.model(input_ids, attention_mask=attention_mask)

                sentence_embeddings = self.__mean_pooling(model_output, attention_mask) if self.task != "nsp" else model_output[1]
                result += sentence_embeddings

            return torch.stack(result) if convert_to_tensor else result


        @staticmethod
        def __mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # TODO: do evaluation while evaluation
        print("Start evaluation")
        self.evaluator(
            self.eval_model,
            output_path=args.output_dir,
            epoch=state.epoch if state.epoch is not None else -1,
            steps=state.global_step,
        )
        print("End evaluation")
