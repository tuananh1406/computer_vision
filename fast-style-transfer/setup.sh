#! /bin/bash
[ ! -d data ] && mkdir -p data
cd data
[ ! -f imagenet-vgg-verydeep-19.mat ] && wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
[ ! -d bin ] && mkdir -p bin
[ ! -f train2014.zip ] && while true; do 
wget -T 15 -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip && break;
done;
unzip -q train2014.zip
