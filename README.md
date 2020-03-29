# Financial Analytics Classifier

![Usage example](src/resources/readme/usage_example.gif?raw=true)

## Run
1. Copy '.env.example' to '.env' and initialize all properties
2. Run command:
    ```bash
    make start
    ```
> **Note:** Only one interaction is required (other actions are fully automated) -- 
log in to keteride@gmail account to connect to Google Drive. Please wait for the 
"Configuring connection to Google Drive..." step and follow the instractions below.

## Install (e.g. to run only research Jutyper Notebooks)
1. Copy '.env.example' to '.env'
2. Run command:
    ```bash
    make install
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
