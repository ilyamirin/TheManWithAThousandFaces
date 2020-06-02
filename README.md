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

## Train models
1. Upload dataset to keteride@gmail's Google Drive as 'dataset.csv' file with columns 
'object', 'project', 'financing', 'nomenclature' and 'description'.
2. Run command:
    ```bash
    make train-models
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
