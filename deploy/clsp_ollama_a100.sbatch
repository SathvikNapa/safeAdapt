#!/bin/bash
#SBATCH --job-name=safeadapt-ollama-a100
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/snapaug1/safeadapt_ollama_a100.log

# Same node-local-tmpfs approach as clsp_ollama.sbatch (see that file's
# comment) — NFS writes stalled under load, so Ollama + models install
# fresh into /dev/shm on whichever node this lands on, every job (re)start.
set -x
INSTALL_DIR="/dev/shm/snapaug1-ollama-${SLURM_JOB_ID}"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"
curl -sL --max-time 300 https://github.com/ollama/ollama/releases/download/v0.40.13/ollama-linux-amd64.tar.zst | unzstd | tar -xf -

export OLLAMA_MODELS="/dev/shm/snapaug1-ollama-${SLURM_JOB_ID}-models"
mkdir -p "$OLLAMA_MODELS"

echo "=== nvidia-smi at job start ==="
nvidia-smi

"$INSTALL_DIR/bin/ollama" serve &
SERVER_PID=$!

echo "=== waiting for ollama server ==="
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:11434/api/version; then
        echo "server is up"
        break
    fi
    sleep 2
done

echo "=== pulling models ==="
"$INSTALL_DIR/bin/ollama" pull qwen2.5:7b-instruct
"$INSTALL_DIR/bin/ollama" pull qwen2.5:14b-instruct

echo "=== models ready ==="
"$INSTALL_DIR/bin/ollama" list

wait "$SERVER_PID"
