import os
import pandas
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from core.turnover_model import TurnoverModel


RESOURCES_PATH = 'src/resources/production'


model = TurnoverModel.build_untrained(pandas.read_csv(f'{RESOURCES_PATH}/dataset.csv'))
model.fit()
model.save()
