run:
    mkdir -p output
    python player.py

check:
    ruff check *.py
    mypy --strict *.py

test:
    pytest

fmt:
    ruff check . --select=I001 --fix
    ruff format .
