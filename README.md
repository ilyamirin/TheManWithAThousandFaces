# Financial Analytics Classifier

![Usage example](src/resources/readme/usage_example.gif?raw=true)

## Run
1. Copy '.env.example' to '.env' and initialize all properties
2. Run command:
    ```bash
    make start
    ```
> **Note 1:** Only one interaction is required (other actions are fully automated) -- 
log in to keteride@gmail account to connect to Google Drive. Please wait for the 
"Configuring connection to Google Drive..." step and follow the instractions below.

> **Note 2:** The first launch takes from 15 to 30 minutes due to the need to download
and process more than 10 GB of data. But subsequent launches take less than 1 minute.

## Retrain models
1. Prepare a dataset patch as the CSV file with columns 'ЦФО', 'ЦФОГУИД', 'ВЦСШапка', 'ВЦСШапка ГУИД', 'Ви пи проект шапка', 'Ви пи проект шапка ГУИД', 'Номенклатура', 'Номенклатура ГУИД', 'Характеристика номенклатуры', 'Код ОКВЭД', 'Код ОКВЭДГУИД', 'Код ОКПД', 'Код ОКПДГУИД', 'ВЦСТаблица ДЭП', 'ВЦСТаблица ДЭПГУИД', 'Ви пи проект таблица ДЭП', 'Ви пи проект таблица ДЭПГУИД', 'Мероприятие', 'Мероприятие ГУИД', 'Статья оборотов', 'Статья оборотов ГУИД', 'Смета', 'Смета ГУИД', 'КВР', 'КВРГУИД'.
2. Run command:
    ```bash
    make retrain-models DATASET_PATCH=<path_to_csv_file>
    ```

## Install for Development
1. Copy '.env.example' to '.env'
2. Run command:
    ```bash
    make init-local-dev
    ```

## Other Commands
* Show logs (of 'ui' or 'app')
    ```bash
    make logs app
    make logs ui
    ```
* Restart (with rebuild):
    ```bash
    make restart
    ```
* Restart (without rebuild):
    ```bash
    make run
    ```
