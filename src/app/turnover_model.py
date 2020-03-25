import re
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from embedding import fasttext_model
from domain import TurnoverInput, NetOutput, Prediction


RESOURCES_PATH = 'src/resources'


class TurnoverModel:
    _model: Model
    _fasttext = fasttext_model()
    _budget_le: LabelEncoder

    _MAX_NOMENCLATURE_LEN = 17
    _MAX_DESCRIPTION_LEN = 30


    @classmethod
    def load_trained(cls):
        slf = cls()
        slf._model = slf._load_trained_model()
        slf._budget_le = slf._load_label_encoder('budgets')
        slf._warm_up()
        return slf
    
    def predict(self, inp: TurnoverInput) -> NetOutput:
        x_vec = self._to_x_vec(inp.nomenclature, inp.description)
        y_pred_vec = self._model.predict(x_vec)
        return self._to_net_output(y_pred_vec)

    def _to_x_vec(self, nomenclature: str, description: str):
        nom_vec = pad_sequences([self._get_embeddings(nomenclature)], maxlen=self._MAX_NOMENCLATURE_LEN, dtype='float32')
        desc_vec = pad_sequences([self._get_embeddings(description)], maxlen=self._MAX_DESCRIPTION_LEN, dtype='float32')
        return nom_vec, desc_vec
    
    def _to_net_output(self, y_pred_vec: np.ndarray) -> NetOutput:
        pred_sorted = sorted(enumerate(y_pred_vec[0]), key=lambda i: i[1], reverse=True)
        main_pred = pred_sorted[0]
        alt_preds = pred_sorted[1:4]
        return NetOutput(
            self._to_prediction(main_pred[0], main_pred[1]), 
            [self._to_prediction(i[0], i[1]) for i in alt_preds]
        )
    
    def _to_prediction(self, target_i: int, prob: float) -> Prediction:
        return Prediction(
            self._budget_le.inverse_transform([target_i])[0],
            round(prob * 100, 2)
        )

    def _get_embeddings(self, phrase: str) -> np.ndarray:
        phrase_tokens = self._clear_phrase(phrase).split()
        return np.array(list(map(self._fasttext.get_word_vector, phrase_tokens)))
    
    def _clear_phrase(self, phrase: str) -> str:
        lower_cased = phrase.lower()
        without_special_chars = re.sub(r"[^a-zА-я0-9 ]", '', lower_cased)
        without_excess_spaces = re.sub(r" {2,}", ' ', without_special_chars)
        stripped = without_excess_spaces.strip()
        return stripped

    def _load_trained_model(self) -> Model:
        print('Loading turnover model...')
        model = load_model(f'{RESOURCES_PATH}/production/turnover/model.h5')
        print('├── Complete')
        return model
    
    def _load_label_encoder(self, name: str) -> LabelEncoder:
        le = LabelEncoder()
        le.classes_ = np.array(Path(f'{RESOURCES_PATH}/production/turnover/{name}.txt').read_text().split('\n'))
        return le
    
    def _warm_up(self) -> None:
        print('Warming up turnover model...')
        self.predict(TurnoverInput('Warm up', 'Warm up'))
        print('├── Complete')
