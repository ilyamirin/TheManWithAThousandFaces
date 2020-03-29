include .env

REACT_APP_API_URL=http://${SERVER_IP}:${APP_PORT}

export


args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

%:
    @:


install:
	@./src/operation/resolve-system-dependencies.sh
	@./src/operation/resolve-app-dependencies.sh

update-python-libs:
	@./.venv/bin/pip install -r requirements.txt

config-drive-connection:
	@./src/operation/config-drive-connection.sh

sync-resources-inbound: config-drive-connection
	@./src/operation/sync-resources-inbound.sh

train-turnover-model:
	@.venv/bin/python3.6 src/app/train_turnover_model.py
	@./src/operation/sync-resources-outbound.sh

train-budget-model:
	@.venv/bin/python3.6 src/app/train_budget_model.py
	@./src/operation/sync-resources-outbound.sh

train-models: train_budget_model train_turnover_model

build:
	@./src/operation/build.sh

run:
	@./src/operation/run.sh

stop:
	@./src/operation/stop.sh

start: install sync-resources-inbound build run
	@echo "Successfully Started"

restart: stop start

logs:
	@docker logs -f fac-$(call args)
