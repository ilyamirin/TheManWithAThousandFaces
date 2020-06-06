include .env

REACT_APP_API_URL=http://${SERVER_IP}:${APP_PORT}

export


args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

%:
    @:


init-system:
	@./src/operation/init-system.sh

init-local-dev: init-system
	@./src/operation/init-local-dev.sh

update-python-libs:
	@./.venv/bin/pip install -r requirements.txt

build: init-system
	@./src/operation/build.sh

run:
	@./src/operation/run.sh

run-train-budget-model:
	@./src/operation/run-train-budget-model.sh

run-train-turnover-model:
	@./src/operation/run-train-turnover-model.sh

run-supplement-dataset:
	@./src/operation/run-supplement-dataset.sh $(DATASET_PATCH)

stop:
	@./src/operation/stop.sh

start: build run
	@echo "Successfully Started"

restart: stop start

train-turnover-model: build run-train-turnover-model

train-budget-model: build run-train-budget-model

train-models: train-turnover-model train-budget-model

logs:
	@docker logs -f fac-$(call args)
