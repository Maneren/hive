run:
    mkdir -p output
    poetry run hive

check:
    poetry run ruff check src/
    poetry run mypy src/

test:
    poetry run pytest

fmt:
    poetry run ruff check . --select=I001 --fix
    poetry run ruff format .