import re
from datetime import datetime
from time import time, strftime, gmtime
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from scipy.special import softmax
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from core.embedding import fasttext_model
from core.domain import BudgetInput, NetOutput, Prediction, BusinessException


RESOURCES_PATH = 'src/resources/production/budget'

EMBEDDING_VEC_LEN = 300
MAX_PHRASE_LEN = 15
FIT_MAX_EPOCHS = 300
FIT_EARLY_STOP_PATIENCE = 30
FIT_VALIDATION_SIZE = 0.1

FINANCING_MASK_MAP = {
    'Ханьбань, КНР':     'Ханьбань, КНР',
    'Дальнефтепровод':   'Дальнефтепровод',
    'Фонд Оксфорда':     'Фонд Оксфорда',
    'ЭАЦ':               'ЭАЦ',
    'УНМ':               'УНМ',
    'Роснефть-классы':   'Роснефть-классы',
    'Д-000-00':          r'^Д-\d{1,3}-\d{2}(_.+)?(\/\d+)?$',
    '00-00-00000':       r'^\d{2}-\d{2,3}-\d{5}(\/\d{2})?(-П)?$',
    '000000/00000Д':     r'^\d{6,7}\/\d{4,5}Д$',
    '0000-0000/000-000': r'^\d{4}-\d{4}\/\d{3}-\d{3}$',
    '000-00Т':           r'^\d{3}\/\d{2}Т$',
    '0000-000-00-0':     r'^\d{4}-\d{3}-\d{2}-\d$',
    'DD-MM-YY':          r'^\d{2}-\d{2}-\d{2}$',
    '0/0000/0000':       r'^\d{1}\/\d{4}\/\d{4}$',
    '00.000.00.0000':    r'^\d{2}\.\d{3}\.\d{2}\.\d{4}$',
    '000-00-0000-000/0': r'^\d{3}-\d{2}-\d{4}-\d{3}\/\d$',
    'МК-0000.0000.0':    r'^МК-\d{4}\.\d{4}\.\d(\/\d{2})?$'
}


class _EmbeddingFFNN(nn.Module):
    def __init__(self, target_len: int, financing_len: int, optimizer_fn=None, loss=None):
        super(_EmbeddingFFNN, self).__init__()

        self.object_linear = nn.Linear(EMBEDDING_VEC_LEN, 512)
        self.object_batch_norm = nn.BatchNorm1d(512)
        self.object_dropout = nn.Dropout(0.2)
        
        self.project_linear = nn.Linear(EMBEDDING_VEC_LEN, 512)
        self.project_batch_norm = nn.BatchNorm1d(512)
        self.project_dropout = nn.Dropout(0.2)

        self.financing_linear = nn.Linear(financing_len, 512)
        self.financing_batch_norm = nn.BatchNorm1d(512)
        self.financing_dropout = nn.Dropout(0.2)

        self.common_linear = nn.Linear(512 * 3, 512)
        self.common_batch_norm = nn.BatchNorm1d(512)
        self.common_dropout = nn.Dropout(0.2)

        self.cls_linear = nn.Linear(512, target_len)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
    
    def forward(self, x: [tensor, tensor, tensor]) -> tensor:
        obj_inp, prj_inp, fin_inp = x[0], x[1], x[2]

        obj_mean_emb = torch.mean(obj_inp, dim=1)
        prj_mean_emb = torch.mean(prj_inp, dim=1)

        obj_branch = self.object_linear(obj_mean_emb)
        obj_branch = self.object_batch_norm(obj_branch)
        obj_branch = F.relu(obj_branch)
        obj_branch = self.object_dropout(obj_branch)

        prj_branch = self.project_linear(prj_mean_emb)
        prj_branch = self.project_batch_norm(prj_branch)
        prj_branch = F.relu(prj_branch)
        prj_branch = self.project_dropout(prj_branch)

        fin_branch = self.financing_linear(fin_inp)
        fin_branch = self.financing_batch_norm(fin_branch)
        fin_branch = F.relu(fin_branch)
        fin_branch = self.financing_dropout(fin_branch)

        concatenated_branches = torch.cat((obj_branch, prj_branch, fin_branch), dim=1)

        common_branch = self.common_linear(concatenated_branches)
        common_branch = self.common_batch_norm(common_branch)
        common_branch = F.relu(common_branch)
        common_branch = self.common_dropout(common_branch)

        logits = self.cls_linear(common_branch)

        return logits
    
    def evaluate(self, x: tensor, y: tensor) -> [float, float]:
        self.eval()

        with torch.no_grad():
            y_pred_logits = self(x)
        
        y_pred_proba = softmax(y_pred_logits.numpy())
        acc = round(accuracy_score(y, y_pred_proba.argmax(axis=1)), 4)
        loss = round(log_loss(y, y_pred_proba, labels=range(y_pred_logits.shape[1])), 4)

        return loss, acc


class _DatasetImpl(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return [i[index] for i in self.x], self.y[index]


class BudgetModel:
    _model: _EmbeddingFFNN
    _fasttext = fasttext_model()
    _budget_le: LabelEncoder
    _financing_le: LabelEncoder
    _df: DataFrame
    _fit_history = []


    @classmethod
    def load_trained(cls):
        slf = cls()
        slf._model = slf._load_trained_model()
        slf._budget_le = slf._load_label_encoder('budget')
        slf._financing_le = slf._load_label_encoder('financing')
        slf._warm_up()
        return slf
    
    @classmethod
    def build_untrained(cls, df: DataFrame):
        slf = cls()
        slf._set_reproducibility()
        slf._df = slf._prepare_dataset(df)
        slf._budget_le = slf._build_label_encoder(slf._df, 'budget')
        slf._financing_le = slf._build_label_encoder(slf._df, 'financing')
        slf._model = _EmbeddingFFNN(len(slf._budget_le.classes_), len(slf._financing_le.classes_))
        return slf
    

    def predict(self, inp: BudgetInput) -> NetOutput:
        self._model.eval()
        x_vec = self._to_x_vec(inp.obj, inp.project, inp.financing)
        with torch.no_grad():
            y_pred_logits = self._model(x_vec)
        return self._to_net_output(y_pred_logits)

    def fit(self):
        Path(f'{RESOURCES_PATH}/categorical').mkdir(parents=True, exist_ok=True)

        x, y = self._get_train_vecs()
        x_train, y_train, x_val, y_val = self._split_to_train_val(x, y)

        best_epoch = 0
        best_loss = 10e100
        started_at = time()

        train_dataloader = DataLoader(_DatasetImpl(x_train, y_train), batch_size=64, shuffle=True)

        for epoch in range(1, FIT_MAX_EPOCHS):
            train_losses = []

            for x, y in train_dataloader:
                self._model.train()

                y_pred = self._model(x)

                loss = self._model.loss(y_pred, y)
                loss.backward()

                self._model.optimizer.step()
                self._model.optimizer.zero_grad()

                train_losses.append(float(loss))
            
            val_loss, val_acc = self._model.evaluate(x_val, y_val)
            train_loss = np.array(train_losses).mean()

            self._fit_history.append({
                'Validation Accuracy': val_acc,
                'Validation Loss': val_loss,
                'Train Loss': train_loss
            })

            print(f'Epoch #{epoch}: Val. Loss -- {val_loss}, Val. Accuracy -- {val_acc}, Train Loss -- {train_loss}, Spent time -- {strftime("%Hh %Mm %Ss", gmtime(time() - started_at))}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(self._model, f'{RESOURCES_PATH}/model.pt')
                print(f'    Validation loss has improved to {best_loss}. Model saved')
            elif epoch - best_epoch > FIT_EARLY_STOP_PATIENCE:
                print(f'    Early stop training. Best validation loss - {best_loss} of epoch #{best_epoch}')
                break
            else:
                print(f"    Validation loss hasn't improved. Current best value - {best_loss} of epoch #{best_epoch}")

        self._model = torch.load(f'{RESOURCES_PATH}/model.pt')

        loss, acc = self._model.evaluate(x_val, y_val)
        print(f'Training completed. Final accuracy = {round(acc, 4)}, loss = {round(loss, 4)}')
    
    def save(self):
        torch.save(self._model, f'{RESOURCES_PATH}/model.pt')
        pd.DataFrame(self._fit_history).to_csv(f'{RESOURCES_PATH}/history.tsv', index=False, sep='\t')
        with open(f'{RESOURCES_PATH}/categorical/budget.txt', 'w') as fout: 
            print(*self._budget_le.classes_, sep='\n', end='', file=fout)
        with open(f'{RESOURCES_PATH}/categorical/financing.txt', 'w') as fout: 
            print(*self._financing_le.classes_, sep='\n', end='', file=fout)


    def _to_x_vec(self, obj: str, project: str, financing: str) -> [tensor, tensor, tensor]:
        obj_vec = pad_sequences([self._get_embeddings(obj)], maxlen=MAX_PHRASE_LEN, dtype='float32')
        prj_vec = pad_sequences([self._get_embeddings(project)], maxlen=MAX_PHRASE_LEN, dtype='float32')
        fin_vec = to_categorical(self._financing_le.transform([self._mask_financing(financing, raise_error=True)]), len(self._financing_le.classes_))
        return tensor(obj_vec), tensor(prj_vec), tensor(fin_vec)

    def _get_train_vecs(self) -> [[tensor, tensor, tensor], tensor]:
        obj_embeddings = [self._get_embeddings(i) for i in self._df.object]
        prj_embeddings = [self._get_embeddings(i) for i in self._df.project]

        obj_vec = pad_sequences(obj_embeddings, maxlen=MAX_PHRASE_LEN, dtype='float32')
        prj_vec = pad_sequences(prj_embeddings, maxlen=MAX_PHRASE_LEN, dtype='float32')
        fin_vec = to_categorical(self._financing_le.transform(self._df.financing), len(self._financing_le.classes_))
        budget_vec = self._budget_le.transform(self._df.budget)

        return [tensor(obj_vec), tensor(prj_vec), tensor(fin_vec)], tensor(budget_vec)
    
    def _split_to_train_val(self, x: [tensor, tensor, tensor], y: tensor) -> [[tensor, tensor, tensor], tensor, [tensor, tensor, tensor], tensor]:
        skf = StratifiedKFold(int(1/FIT_VALIDATION_SIZE), shuffle=True, random_state=42)
        train_i, val_i = next(skf.split(x[0], y))
        x_train, y_train = [x[0][train_i], x[1][train_i], x[2][train_i]], y[train_i]
        x_val, y_val = [x[0][val_i], x[1][val_i], x[2][val_i]], y[val_i]
        return x_train, y_train, x_val, y_val

    def _to_net_output(self, y_pred_logits: tensor) -> NetOutput:
        y_pred_proba = softmax(y_pred_logits.numpy())
        pred_sorted = sorted(enumerate(y_pred_proba[0]), key=lambda i: i[1], reverse=True)
        main_pred = pred_sorted[0]
        alt_preds = pred_sorted[1:4]
        return NetOutput(
            self._to_prediction(main_pred[0], main_pred[1]), 
            [self._to_prediction(i[0], i[1]) for i in alt_preds]
        )

    def _mask_financing(self, value: str, raise_error = False):
        if str(value) == 'nan' or value == '':
            return ''

        for k in FINANCING_MASK_MAP:
            if re.match(FINANCING_MASK_MAP[k], value):
                return k

        if raise_error:
            raise BusinessException('Неподдерживаемый формат ВЦС')
        else:
            return 'UNKNOWN'

    def _get_embeddings(self, phrase: str) -> np.ndarray:
        phrase_tokens = self._clear_phrase(phrase).split()
        if (len(phrase_tokens) == 0): return np.array([np.zeros(EMBEDDING_VEC_LEN)])
        else: return np.array(list(map(self._fasttext.get_word_vector, phrase_tokens)))
    
    def _clear_phrase(self, phrase: str) -> str:
        lower_cased = phrase.lower()
        without_special_chars = re.sub(r"[^a-zА-я0-9 ]", '', lower_cased)
        without_excess_spaces = re.sub(r" {2,}", ' ', without_special_chars)
        stripped = without_excess_spaces.strip()
        return stripped

    def _to_prediction(self, target_i: int, prob: float) -> Prediction:
        return Prediction(
            self._budget_le.inverse_transform([target_i])[0],
            round(prob * 100, 2)
        )

    def _load_trained_model(self) -> _EmbeddingFFNN:
        print('Loading budget model...')
        model = torch.load(f'{RESOURCES_PATH}/model.pt')
        model.eval()
        print('├── Complete')
        return model
    
    def _load_label_encoder(self, label: str) -> LabelEncoder:
        le = LabelEncoder()
        le.classes_ = np.array(Path(f'{RESOURCES_PATH}/categorical/{label}.txt').read_text().split('\n'))
        return le
    
    def _build_label_encoder(self, df: DataFrame, label: str) -> LabelEncoder:
        le = LabelEncoder()
        le.fit(df[label])
        return le

    def _warm_up(self) -> None:
        print('Warming up budget model...')
        self.predict(BudgetInput('Warm up', 'Warm up', 'Д-123-45'))
        print('├── Complete')

    def _set_reproducibility(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
    
    def _prepare_dataset(self, df: DataFrame) -> DataFrame:
        print('Preparing dataset...')
        df = df[['object', 'financing', 'project', 'budget']]
        df = df.drop(df[df.budget.isnull()].index)
        df.financing = df.financing.replace('БЕЗ ВЦС', np.NaN)
        df = df.fillna('')
        df = self._replace_year_specific_targets(df)
        df = self._mask_financing_df(df)
        df = self._extract_unique_dataset(df)
        df = self._remove_rare_targets(df)
        print('├── Complete')
        return df

    def _replace_year_specific_targets(self, df: DataFrame) -> DataFrame:
        current_year = datetime.now().year - 2000
        fixed_df = df
        for i in range(10, current_year):
            fixed_df.budget = fixed_df.budget.replace(f'Ппкс 20{i}', f'Ппкс 20{current_year}')
            fixed_df.budget = fixed_df.budget.replace(f'Субсидия на ИЦ_ОЗОБ 20{i}', f'Субсидия на ИЦ_ОЗОБ 20{current_year}')
        return fixed_df
    
    def _mask_financing_df(self, df: DataFrame) -> DataFrame:
        fixed_df = df
        fixed_df.financing = fixed_df.financing.apply(self._mask_financing)
        fixed_df = fixed_df.drop(fixed_df[fixed_df.financing == 'UNKNOWN'].index)
        return fixed_df

    def _extract_unique_dataset(self, df: DataFrame) -> DataFrame:
        partly_unique_df = df.groupby(['object', 'financing', 'project', 'budget'])\
            .size().reset_index().rename(columns={0:'count'})
        unique_df = partly_unique_df.groupby(['object', 'financing', 'project'], as_index=False)\
            .apply(lambda x: x[x['count'] == x['count'].max()]).reset_index(drop=True)
        return unique_df
    
    def _remove_rare_targets(self, df: DataFrame) -> DataFrame:
        rare_budget_df = df.groupby('budget').agg({'count': ['count', 'sum']})
        rare_budget_df.columns = ['count', 'original_count']
        rare_budgets = rare_budget_df[(rare_budget_df['count'] < 3) & (rare_budget_df['original_count'] < 150)].index
        cleared_df = df[~df.budget.isin(rare_budgets)]
        return cleared_df
