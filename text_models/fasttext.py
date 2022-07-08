from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

from data_processing.util import Section
from text_models.abstract_model import AbstractModel


class FastTextModel(AbstractModel):
    """
    FastText model, that can be fine-tuned on docs and can also be used to map sentences / text to embeddings.

    :param vector_size: The size of embedding vector
    :param epochs: The number of train epochs
    :param min_count: Ignores all words with total frequency lower than this
    :param pretrained_model: The name of pretrained text model
    :param seed: Random seed
    :param save_to_path: Where the trained model should be saved
    """

    def __init__(
        self,
        vector_size: int = 300,
        epochs: int = 5,
        min_count: int = 1,
        pretrained_model: str = "undefined",
        seed: int = 42,
        save_to_path: str = "./",
    ):
        super().__init__(vector_size, epochs, pretrained_model, seed, save_to_path)
        self.min_count = min_count

    name = "FastText"

    def train_task(self, corpus: Section):
        self.model = FastText(corpus, vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)

    def train_pt_task(self, corpus: Section):
        self.model = load_facebook_model(self.pretrained_model)
        self.model.build_vocab(corpus, update=True)
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    @classmethod
    def load(cls, path: str):
        loaded_model = FastText.load(path)
        created_model = FastTextModel(loaded_model.vector_size, loaded_model.epochs, loaded_model.min_count)
        created_model.model = loaded_model.wv
        return created_model
