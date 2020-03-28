include .env

REACT_APP_API_URL=http://${SERVER_IP}:${API_PORT}

export


args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

%:
    @:


resolve-dependencies:
	@./src/operation/resolve-system-dependencies.sh
	@./src/operation/resolve-app-dependencies.sh

update-python-libs:
	@./.venv/bin/pip install -r requirements.txt

config-drive-connection:
	@./src/operation/config-rclone.sh

sync-resources-inbound:
	@rclone sync KeterideDrive:Financial-Analytics-Classifier src/resources/production

sync-resources-outbound:
	@rclone sync src/resources/production KeterideDrive:Financial-Analytics-Classifier

train-turnover-model:
	@.venv/bin/python3.6 src/app/train_turnover_model.py
	@rclone sync src/resources/production KeterideDrive:Financial-Analytics-Classifier
