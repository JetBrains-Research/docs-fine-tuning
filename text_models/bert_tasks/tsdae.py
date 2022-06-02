import os.path
from typing import List

from sentence_transformers import models, losses, SentenceTransformer, evaluation
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

from data_processing.util import sections_to_sentences
from text_models.bert_tasks import AbstractTask


class TSDenoisingAutoEncoderTask(AbstractTask):
    name = "tsdae"

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: List[List[List[str]]],
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:
        corpus = sections_to_sentences(docs_corpus)

        word_embedding_model = models.Transformer(pretrained_model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        if self.n_examples == "all":
            self.n_examples = len(corpus)

        train_dataset = DenoisingAutoEncoderDataset(corpus[: self.n_examples])
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        train_loss = losses.DenoisingAutoEncoderLoss(
            model, decoder_name_or_path=pretrained_model, tie_encoder_decoder=True
        )

        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")
        output_path = os.path.join(save_to_path, "output_docs")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            weight_decay=0,
            scheduler="constantlr",
            optimizer_params={"lr": 3e-5},
            show_progress_bar=True,
            evaluator=evaluator,
            evaluation_steps=self.eval_steps,
            checkpoint_path=checkpoints_path,
            output_path=output_path,
            checkpoint_save_total_limit=3,
            save_best_model=self.save_best_model,
        )

        if self.save_best_model:
            model = SentenceTransformer(output_path)

        return model._first_module()
