#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python -m ruff check backend indicmeet tests scripts
python -m pytest -q --tb=short
python scripts/validate_contract.py fixtures
npm --prefix frontend ci
npm --prefix frontend run build
