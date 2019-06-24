#!/bin/sh
source ~/dl/bin/activate
echo "Batch size 64"
python pytorch_main.py --epochs 70 --emb 50 --hid 50 --bs 64
echo ""
python pytorch_main.py --epochs 70 --emb 128 --hid 256 --bs 64
echo ""
python pytorch_main.py --epochs 70 --emb 128 --hid 500 --bs 64

echo "Batch size 10"
python pytorch_main.py --epochs 100 --emb 50 --hid 50 --bs 10
echo ""
python pytorch_main.py --epochs 100 --emb 128 --hid 256 --bs 10
echo ""
python pytorch_main.py --epochs 100 --emb 128 --hid 500 --bs 10

