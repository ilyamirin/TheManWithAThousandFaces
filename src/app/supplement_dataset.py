import sys
import pandas as pd
import os


RESOURCES_PATH = 'src/resources/production'


if (len(sys.argv) != 2):
    print('[ERROR] The first argument with the path to CSV file is required', file=sys.stderr)
    exit()

if (not os.path.isfile(sys.argv[1])):
    print('[ERROR] The CSV file doesn\'t exists', file=sys.stderr)
    exit()

cur_df = pd.read_csv(f'{RESOURCES_PATH}/dataset.csv')
new_df = pd.read_csv(sys.argv[1])

rename_columns_map = {
    'ЦФО': 'object',
    'ЦФОГУИД': 'object_guid',
    'ВЦСШапка': 'financing',
    'ВЦСШапка ГУИД': 'financing_guid',
    'Ви пи проект шапка': 'project',
    'Ви пи проект шапка ГУИД': 'project_guid',
    'Номенклатура': 'nomenclature',
    'Номенклатура ГУИД': 'nomenclature_guid',
    'Характеристика номенклатуры': 'description',
    'Код ОКВЭД': 'code1',
    'Код ОКВЭДГУИД': 'code1_guid',
    'Код ОКПД': 'code2',
    'Код ОКПДГУИД': 'code2_guid',
    'ВЦСТаблица ДЭП': '_financing',
    'ВЦСТаблица ДЭПГУИД': '_financing_guid',
    'Ви пи проект таблица ДЭП': '_project',
    'Ви пи проект таблица ДЭПГУИД': '_project_guid',
    'Мероприятие': 'event',
    'Мероприятие ГУИД': 'event_guid',
    'Статья оборотов': 'turnover',
    'Статья оборотов ГУИД': 'turnover_guid',
    'Смета': 'budget',
    'Смета ГУИД': 'budget_guid',
    'КВР': 'code3',
    'КВРГУИД': 'code3_guid'
}

new_df.columns = list(map(lambda i: rename_columns_map[i], new_df.columns))

merged_df = pd.concat([cur_df, new_df])
merged_df.to_csv(f'{RESOURCES_PATH}/dataset.csv', index=False)
