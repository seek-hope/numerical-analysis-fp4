#!/bin/bash
# remote_python.sh — Run a Python script on the remote server with GPU allocation.
# Usage: ./remote_python.sh <script_path> [args...]
# Run from the project root.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SSHPASS_FILE="$PROJECT_ROOT/.sshpass"

if [ $# -eq 0 ]; then
    echo "Usage: ./remote_python.sh <script_path> [additional_args...]"
    echo "Example: ./remote_python.sh src/track_a/train.py"
    exit 1
fi

SCRIPT="$1"
shift

sshpass -f "$SSHPASS_FILE" ssh bi_group2@lulab_4090 \
    "cd /home/bi_group2/Projects/Numerical_Analysis && export PYTHONPATH=/home/bi_group2/Projects/Numerical_Analysis:\$PYTHONPATH && conda activate sle && python $SCRIPT $*"
