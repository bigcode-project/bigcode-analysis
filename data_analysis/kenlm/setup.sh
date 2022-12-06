sudo apt -y install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
cd kenlm
python setup.py install