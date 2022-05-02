import tempfile
from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import models
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertTokenizerFast


class AbstractTask(ABC):
    def __init__(self, epochs=2, batch_size=16, n_examples="all", max_len=512, name="abstract"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.name = name
        self.n_examples = n_examples

    @abstractmethod
    def create_model_from_scratch(self, train_sentences: List[str], tmp_file: str):
        raise NotImplementedError()

    @abstractmethod
    def finetune_on_docs(
        self, pretrained_model: str, docs_corpus: List[str], device: str, save_to_path: str
    ) -> models.Transformer:
        raise NotImplementedError()

    def _create_and_train_tokenizer(
        self, train_sentences: List[str], vocab_size: int, tmp_file: str
    ) -> BertTokenizerFast:
        with open(tmp_file, "w") as fp:
            fp.write("\n".join(train_sentences))

        tokenizer = BertWordPieceTokenizer(
            clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True
        )
        tokenizer.train(
            files=[tmp_file],
            vocab_size=vocab_size,
            min_frequency=1,
            wordpieces_prefix="##",
            special_tokens=["[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]"],
        )

        save_to_path = tempfile.mkdtemp()

        tokenizer.save_model(save_to_path)

        return BertTokenizerFast.from_pretrained(save_to_path, max_len=self.max_len + 2)
