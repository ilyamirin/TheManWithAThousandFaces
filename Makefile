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
