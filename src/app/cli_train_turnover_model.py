import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from turnover_model import TurnoverModel


RESOURCES_PATH = 'src/resources'


model = TurnoverModel.build_untrained(f'{RESOURCES_PATH}/dataset/original.csv')
model.fit()
model.save()
