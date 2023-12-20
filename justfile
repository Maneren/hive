run:
    mkdir -p output
    python player.py

check:
    ruff check src/
    mypy src/

test:
    pytest

fmt:
    ruff check . --select=I001 --fix
    ruff format .