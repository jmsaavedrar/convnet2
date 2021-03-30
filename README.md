# convnet2
cnn based on tensorflow 2.3
# Using Checkpoints
You can use a checkpoint to initialize the model with pre-trained weights. To this end, you will need  to set the chekpoint file using the parameter **CKPFILE** in the corresponding configuration file.
# Checkpoints 
The following checpoints were produced by a ResNet-34
## ImageNet
[Download](https://www.dropbox.com/s/ea61crvnckf96ez/imagenet_045.h5)

md5sum: a456fe88f2bad870b2218661848169d0  

## Sketches

[Download](https://www.dropbox.com/s/kb443ulitvipixy/sketch_050.h5)

md5sum: a53f18d41b2b3b4c4dc8ce5026c6317c
# Datasets
## MNIST-5000
[Download](https://www.dropbox.com/s/abi61g7adjdbmih/MNIST-5000.zip)
## MNIST-FULL
- Training Images [60000] [download](https://www.dropbox.com/s/6lmn4fre326cty2/mnist_test.gzip)
- Testing Images [60000]  [download](https://www.dropbox.com/s/knvoss1iukj42pk/mnist_train.gzip)
A siimple code you can run to genereate test.txt and train.txt
```
find $(pwd)/Test -name *.png | awk -F '\/' '{f=$NF; sub(".png","",f); gsub(".*_","",f);print $0"\t"f}' > test.txt
```
## Sketches-Eitz
[Download]https://www.dropbox.com/s/ut350iwgby9swk2/Sketch_EITZ.zip?dl=0

For more details visit http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
## Sketches-QuickDraw
https://github.com/googlecreativelab/quickdraw-dataset
## Sketches-QuickDraw (sample Animals )
[Download](https://www.dropbox.com/sh/hsqjv0kd13xda3g/AABYkVk0ruG85s4aL4C1nDKaa)



