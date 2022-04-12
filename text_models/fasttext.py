from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

from text_models.abstract_model import AbstractModel


class FastTextModel(AbstractModel):
    def __init__(self, vector_size=300, epochs=5, min_count=1, pretrained_model=None, seed=42):
        super().__init__(vector_size, epochs, pretrained_model, seed)
        self.min_count = min_count

    name = "ft"

    def train_from_scratch(self, corpus):
        self.model = FastText(corpus, vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)

    def train_pretrained(self, corpus):
        self.model = load_facebook_model(self.pretrained_model)
        self.model.build_vocab(corpus, update=True)
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    @classmethod
    def load(cls, path):
        loaded_model = FastText.load(path)
        created_model = FastTextModel(loaded_model.vector_size, loaded_model.epochs, loaded_model.min_count)
        created_model.model = loaded_model.wv
        return created_model
