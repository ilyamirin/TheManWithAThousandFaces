import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Lambda, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from core.embedding import fasttext_model
from core.domain import TurnoverInput, NetOutput, Prediction


RESOURCES_PATH = 'src/resources/production/turnover'

EMBEDDING_VEC_LEN = 300
MAX_NOMENCLATURE_LEN = 17
MAX_DESCRIPTION_LEN = 30
FIT_MAX_EPOCHS = 300
FIT_EARLY_STOP_PATIENCE = 30
FIT_VALIDATION_SIZE = 0.2


class TurnoverModel:
    _model: Model
    _fasttext = fasttext_model()
    _turnover_le: LabelEncoder
    _df: DataFrame
    _fit_history: History


    @classmethod
    def load_trained(cls):
        slf = cls()
        slf._model = slf._load_trained_model()
        slf._turnover_le = slf._load_label_encoder('turnover')
        slf._warm_up()
        return slf
    
    @classmethod
    def build_untrained(cls, dataset_path: str):
        slf = cls()
        slf._set_reproducibility()
        slf._df = slf._prepare_dataset(pd.read_csv(dataset_path))
        slf._turnover_le = slf._build_label_encoder(slf._df, 'turnover')
        slf._model = slf._build_untrained_model(len(slf._turnover_le.classes_))
        return slf

    
    def predict(self, inp: TurnoverInput) -> NetOutput:
        x_vec = self._to_x_vec(inp.nomenclature, inp.description)
        y_pred_vec = self._model.predict(x_vec)
        return self._to_net_output(y_pred_vec)
    
    def fit(self):
        Path(f'{RESOURCES_PATH}/categorical').mkdir(parents=True, exist_ok=True)

        x, y = self._get_train_vecs()
        x_train, y_train, x_val, y_val = self._split_to_train_val(x, y)

        self._fit_history = self._model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=FIT_MAX_EPOCHS,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=FIT_EARLY_STOP_PATIENCE),
                ModelCheckpoint(f'{RESOURCES_PATH}/model.h5', monitor='val_loss', save_best_only=True, verbose=1)
            ],
            verbose=1
        )

        self._model = load_model(f'{RESOURCES_PATH}/model.h5')

        loss, acc = self._model.evaluate(x_val, y_val)
        print(f'Training completed. Final accuracy = {round(acc, 4)}, loss = {round(loss, 4)}')
    
    def save(self):
        self._model.save(f'{RESOURCES_PATH}/model.h5')

        pd.DataFrame({
            'train_loss': self._fit_history.history['loss'], 
            'val_loss': self._fit_history.history['val_loss'], 
            'val_acc': self._fit_history.history['val_accuracy']
            })\
            .to_csv(f'{RESOURCES_PATH}/history.tsv', index=False, sep='\t')
        
        with open(f'{RESOURCES_PATH}/categorical/turnover.txt', 'w') as fout: 
            print(*self._turnover_le.classes_, sep='\n', file=fout)

    def _to_x_vec(self, nomenclature: str, description: str) -> np.ndarray:
        nom_vec = pad_sequences([self._get_embeddings(nomenclature)], maxlen=MAX_NOMENCLATURE_LEN, dtype='float32')
        desc_vec = pad_sequences([self._get_embeddings(description)], maxlen=MAX_DESCRIPTION_LEN, dtype='float32')
        return nom_vec, desc_vec
    
    def _get_train_vecs(self) -> List[np.ndarray]:
        nom_embeddings = [self._get_embeddings(i) for i in self._df.nomenclature]
        desc_embeddings = [self._get_embeddings(i) for i in self._df.description]

        nom_vec = pad_sequences(nom_embeddings, maxlen=MAX_NOMENCLATURE_LEN, dtype='float32')
        desc_vec = pad_sequences(desc_embeddings, maxlen=MAX_DESCRIPTION_LEN, dtype='float32')
        turnover_vec = to_categorical(self._turnover_le.transform(self._df.turnover))

        return [nom_vec, desc_vec], turnover_vec
    
    def _split_to_train_val(self, x: List[np.ndarray], y: np.ndarray) -> List[np.ndarray]:
        skf = StratifiedKFold(int(1/FIT_VALIDATION_SIZE), shuffle=True, random_state=42)
        train_i, val_i = next(skf.split(x[0], y.argmax(axis=1)))
        x_train, y_train = [x[0][train_i], x[1][train_i]], y[train_i]
        x_val, y_val = [x[0][val_i], x[1][val_i]], y[val_i]
        return x_train, y_train, x_val, y_val
    
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
            self._turnover_le.inverse_transform([target_i])[0],
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
        model = load_model(f'{RESOURCES_PATH}/model.h5')
        print('├── Complete')
        return model
    
    def _load_label_encoder(self, label: str) -> LabelEncoder:
        le = LabelEncoder()
        le.classes_ = np.array(Path(f'{RESOURCES_PATH}/categorical/{label}.txt').read_text().split('\n'))
        return le
    
    def _warm_up(self) -> None:
        print('Warming up turnover model...')
        self.predict(TurnoverInput('Warm up', 'Warm up'))
        print('├── Complete')
    
    def _prepare_dataset(self, df: DataFrame) -> DataFrame:
        print('Preparing dataset...')
        df = df[['nomenclature', 'description', 'turnover']]
        df = df.drop(df[df.turnover.isnull()].index)
        df = df.fillna('')
        df = self._extract_unique_dataset(df)
        df = self._remove_rare_targets(df)
        print('├── Complete')
        return df
        
    def _extract_unique_dataset(self, df: DataFrame) -> DataFrame:
        partly_unique_df = df.groupby(['nomenclature', 'description', 'turnover'])\
            .size().reset_index().rename(columns={0:'count'})
        unique_df = partly_unique_df.groupby(['nomenclature', 'description'], as_index=False)\
            .apply(lambda x: x[x['count'] == x['count'].max()]).reset_index(drop=True)
        return unique_df
    
    def _remove_rare_targets(self, df: DataFrame) -> DataFrame:
        rare_turnover_df = df.groupby('turnover').agg({'count': ['count', 'sum']})
        rare_turnover_df.columns = ['count', 'original_count']
        rare_turnovers = rare_turnover_df[(rare_turnover_df['count'] < 7) & (rare_turnover_df['original_count'] < 150)].index
        cleared_df = df[~df.turnover.isin(rare_turnovers)]
        return cleared_df
    
    def _build_label_encoder(self, df: DataFrame, label: str) -> LabelEncoder:
        le = LabelEncoder()
        le.fit(df[label])
        return le
    
    def _build_untrained_model(self, taget_len: int) -> Model:
        nomenclature_input = Input(shape=(MAX_NOMENCLATURE_LEN, EMBEDDING_VEC_LEN))

        nomenclature_branch = LSTM(64)(nomenclature_input)
        nomenclature_branch = BatchNormalization()(nomenclature_branch)
        nomenclature_branch = Dropout(0.2)(nomenclature_branch)

        description_input = Input(shape=(MAX_DESCRIPTION_LEN, EMBEDDING_VEC_LEN))

        description_branch = LSTM(64)(description_input)
        description_branch = BatchNormalization()(description_branch)
        description_branch = Dropout(0.2)(description_branch)

        common_branch = Concatenate(axis=1)([nomenclature_branch, description_branch])

        common_branch = Dense(512)(common_branch)
        common_branch = BatchNormalization()(common_branch)
        common_branch = Activation("relu")(common_branch)
        common_branch = Dropout(0.2)(common_branch)

        common_branch = Dense(taget_len, activation='softmax')(common_branch)

        model = Model(inputs=[nomenclature_input, description_input], outputs=common_branch)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=5e-5), metrics=['accuracy'])

        return model
    
    def _set_reproducibility(self) -> None:
        np.random.seed(42)
        tf.random.set_seed(42)
