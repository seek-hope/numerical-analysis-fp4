#!/bin/bash
# remote_run.sh — Execute a command on the remote server.
# Usage: ./remote_run.sh "<command>"
# Run from the project root.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SSHPASS_FILE="$PROJECT_ROOT/.sshpass"

if [ $# -eq 0 ]; then
    echo "Usage: ./remote_run.sh '<command>'"
    echo "Example: ./remote_run.sh 'python src/track_a/model.py'"
    exit 1
fi

sshpass -f "$SSHPASS_FILE" ssh bi_group2@bioinfo_class \
    "cd /home/bi_group2/Projects/Numerical_Analysis && export PYTHONPATH=/home/bi_group2/Projects/Numerical_Analysis:\$PYTHONPATH && conda activate sle && $*"
