#!/bin/bash
# sync.sh — Sync the Numerical_Analysis project to the remote server.
# Run from the project root.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
SSHPASS_FILE="$PROJECT_ROOT/.sshpass"
LOCAL_DIR="/home/rimuru/Projects/Code/homework/Numerical_Analysis/proj/"
REMOTE_USER="bi_group2"
REMOTE_HOST="lulab_4090"
REMOTE_DIR="/home/bi_group2/Projects/Numerical_Analysis/"

echo "Syncing $LOCAL_DIR → $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

rsync -avz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='wandb' \
    --exclude='优化数制与运算机制提升模型性能 - Claude*' \
    -e "sshpass -f $SSHPASS_FILE ssh" \
    "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

echo "✓ Sync complete."
