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

pip install -r requirements_vec.txt 
export PYTHONPATH=/root 
