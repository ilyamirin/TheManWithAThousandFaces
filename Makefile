include .env

REACT_APP_API_URL=http://${SERVER_IP}:${APP_PORT}

export


args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

%:
    @:


init-system:
	@./src/operation/init-system.sh

init-google-drive:
	@./src/operation/init-google-drive.sh

init-local-dev: init-system init-google-drive
	@./src/operation/init-local-dev.sh

update-python-libs:
	@./.venv/bin/pip install -r requirements.txt

train-turnover-model:
	@.venv/bin/python3.6 src/app/train_turnover_model.py
	@./src/operation/sync-resources-outbound.sh

train-budget-model:
	@.venv/bin/python3.6 src/app/train_budget_model.py
	@./src/operation/sync-resources-outbound.sh

train-models: train_budget_model train_turnover_model

build: init-system init-google-drive
	@./src/operation/build.sh

run:
	@./src/operation/run.sh

stop:
	@./src/operation/stop.sh

start: build run
	@echo "Successfully Started"

restart: stop start

logs:
	@docker logs -f fac-$(call args)
