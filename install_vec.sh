sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
sudo apt-get install -y python3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
sudo update-alternatives --config python3

curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
pip install setuptools --upgrade
python3 -m pip install -r requirements_vec.txt 
export PYTHONPATH=/root 


pip install --force-reinstall -U setuptools
pip install --force-reinstall -U pip
wget https://bootstrap.pypa.io/ez_setup.py -O - | python

python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tokenizer_mode mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.95 \
  --swap-space 16
  --max-model-len 12000