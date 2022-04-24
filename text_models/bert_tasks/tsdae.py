from typing import List

from sentence_transformers import models, losses, SentenceTransformer
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from transformers import BertConfig, BertModel

from torch.utils.data import DataLoader

from data_processing.util import get_corpus_properties
from text_models.bert_tasks import AbstractTask


class TSDenoisingAutoEncoderTask(AbstractTask):
    def __init__(self, epochs=2, batch_size=16, max_len=512):
        super().__init__(epochs, batch_size, max_len, name="tsdae")

    def create_model_from_scratch(self, train_sentences: List[str], tmp_file: str):
        vocab_size, _ = get_corpus_properties([sentence.split(" ") for sentence in train_sentences])

        tokenizer = self._create_and_train_tokenizer(train_sentences, vocab_size, tmp_file)

        bert_config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=self.max_len + 2,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        dumb_model = BertModel(bert_config)

        return dumb_model, tokenizer

    def finetune_on_docs(
        self, pretrained_model: str, docs_corpus: List[str], device: str, save_to_path: str
    ) -> models.Transformer:
        word_embedding_model = models.Transformer(pretrained_model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        train_dataset = DenoisingAutoEncoderDataset(docs_corpus)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=pretrained_model, tie_encoder_decoder=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            weight_decay=0,
            scheduler='constantlr',
            optimizer_params={'lr': 3e-5},
            show_progress_bar=True
        )

        return model._first_module()





