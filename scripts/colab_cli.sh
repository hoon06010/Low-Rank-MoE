#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SSL_CERT_FILE=$("$TASK_ROOT/.venv-colab/bin/python" -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE
exec "$TASK_ROOT/.venv-colab/bin/colab" "$@"
