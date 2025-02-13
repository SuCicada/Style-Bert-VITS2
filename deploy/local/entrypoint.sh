#!/bin/bash
set -e

# 检测 GPU 是否可用
while ! nvidia-smi > /dev/null 2>&1; do
    echo "Waiting for GPU to be available..."
    sleep 10
done


/home/peng/miniconda3/bin/conda init bash
echo "Running pre-start command..."
#your-command-here  
git config --global --add safe.directory /app
source /root/.bashrc
export PATH=/home/peng/miniconda3/bin:$PATH
#pip install --force-reinstall -U sumake
#make runuvicorn
exec "$@"
