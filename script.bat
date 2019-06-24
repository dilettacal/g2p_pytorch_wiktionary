@ECHO OFF
ECHO Script for training the model
ECHO Sequenze length 3
ECHO ===============================================
ECHO Batch size 100
python pytorch_main.py --emb 500 --hid 500 --bs 100 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 20
python pytorch_main.py --emb 256 --hid 256 --bs 100 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 20
python pytorch_main.py --emb 128 --hid 128 --bs 100 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 20
ECHO ===============================================
ECHO Batch size 64
python pytorch_main.py --emb 500 --hid 500 --bs 64 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 20
python pytorch_main.py --emb 256 --hid 256 --bs 64 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 20
python pytorch_main.py --emb 128 --hid 128 --bs 64 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 20
ECHO ===============================================
ECHO Batch size 24
python pytorch_main.py --emb 500 --hid 500 --bs 24 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 25
python pytorch_main.py --emb 256 --hid 256 --bs 24 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 25
python pytorch_main.py --emb 128 --hid 128 --bs 24 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 25
ECHO ===============================================
ECHO Sequenze length 2
ECHO ===============================================
ECHO Batch size 100
python pytorch_main.py --emb 500 --hid 500 --bs 100 --att True --file p2p_toy_wiki_de-de_2.csv --epochs 25
python pytorch_main.py --emb 256 --hid 256 --bs 100 --att True --file p2p_toy_wiki_de-de_2.csv --epochs 25
python pytorch_main.py --emb 128 --hid 128 --bs 100 --att True --file p2p_toy_wiki_de-de_2.csv --epochs 25
ECHO ===============================================
ECHO Batch size 64
python pytorch_main.py --emb 500 --hid 500 --bs 64 --att True --file p2p_toy_wiki_de-de_2.csv --epochs 30
python pytorch_main.py --emb 256 --hid 256 --bs 64 --att True --file p2p_toy_wiki_de-de_2.csv --epochs 30
python pytorch_main.py --emb 128 --hid 128 --bs 64 --att True --file p2p_toy_wiki_de-de_2.csv --epochs 30
ECHO ===============================================
ECHO Batch size 24
python pytorch_main.py --emb 500 --hid 500 --bs 24 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 30
python pytorch_main.py --emb 256 --hid 256 --bs 24 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 30
python pytorch_main.py --emb 128 --hid 128 --bs 24 --att True --file p2p_toy_wiki_de-de_3.csv --epochs 30
ECHO ===============================================
