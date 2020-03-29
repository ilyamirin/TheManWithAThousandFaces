import os
import pandas
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from core.budget_model import BudgetModel


RESOURCES_PATH = 'src/resources/production'


model = BudgetModel.build_untrained(pandas.read_csv(f'{RESOURCES_PATH}/dataset.csv'))
model.fit()
model.save()
