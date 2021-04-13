# convnet2
cnn based on tensorflow 2.3+. Please, check the instructions to install it [here](https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101).
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
<a name="datasets"></a>
# Datasets
<a name="mnist5000"></a>
## MNIST-5000
This is a sample of the original MNIST dataset with 5000 images for training.

[Download](https://www.dropbox.com/s/abi61g7adjdbmih/MNIST-5000.zip)
## MNIST-FULL
You can download the complete set of images for training and testing following the next links:
- Training Images [60000] [download](https://www.dropbox.com/s/6lmn4fre326cty2/mnist_test.gzip)
- Testing Images [60000]  [download](https://www.dropbox.com/s/knvoss1iukj42pk/mnist_train.gzip)
A siimple code you can run to genereate test.txt and train.txt
```
find $(pwd)/Test -name *.png | awk -F '\/' '{f=$NF; sub(".png","",f); gsub(".*_","",f);print $0"\t"f}' > test.txt
```
## Sketches-Eitz
This is a sketch dataset wiht 250 classes:

[Download]https://www.dropbox.com/s/ut350iwgby9swk2/Sketch_EITZ.zip?dl=0

For more details visit http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
## Sketches-QuickDraw
https://github.com/googlecreativelab/quickdraw-dataset
## Sketches-QuickDraw (sample Animals )
[Download](https://www.dropbox.com/sh/hsqjv0kd13xda3g/AABYkVk0ruG85s4aL4C1nDKaa)

# Running a Simple Example
## Prepare the dataset for training

We will need the files *train.txt* and *test.txt*. These files should contain the list of images that will be used for training and testing, respectively. Each file should come in a two-column format, the first column is the absolute filename for each image, and the second is the corresponding class (0-indexed). The separator between columns is the tab character.

For this example, we will use the MNIST dataset that can be download as specified [above](#datasets). If you want to try with a smaller dataset, you can use [MNIST-5000](#mnist5000).

## Prepare a configuration file

To facilitate the parameter setting, we include a configuration file with the following parameters:
- NUM_EPOCHS: *It is the number of epochs the training will run.*
- NUM_CLASSES = *It is the number of classes in your problem.*
- BATCH_SIZE = *Size of each batch.*
- VALIDATION_STEPS = *It is the number of iterations required to cover the validation dataset. It is equal to validation_size/batch_size.*
- LEARNING_RATE = *It is the learning rate.*
- SNAPSHOT_DIR = *It is the path where the weights will be stored during training.*
- DATA_DIR = *It is the path where train.txt and test.txt are stored.*
- CHANNELS = *It is the number of channels for the input images.*
- IMAGE_TYPE = *It is a customized name that will define the type of preprocessing applied to the input images.*
- IMAGE_WIDTH = *The target image width.*
- IMAGE_HEIGHT = *The target image height.*
- SHUFFLE_SIZE = *It is used to reserve a fixed memory for shuffling the data.*
- CKPFILE = *It is the abosulte path from initial weights are loaded. It is optional.*

We can include different sets of parameters for various experiments. To make each configuration unique, we have a section name.

An example of a configuration file for MNIST can be found [here](configs/mnist_full.config). 

## Create tfrecords 
An efficient way to store the data is through tfrecords. This allows the model to load the dataset quickly. 
```
python datasets/create_tfrecords.py -type all -config configs/mnist_full.config -name MNIST
```
## Train
```
python train_simple.py  -mode train  -config configs/mnist_full.config -name MNIST
```
## Test
```
python train_simple.py  -mode test  -config configs/mnist_full.config -name MNIST
```
## Predict
```
python train_simple.py  -mode predict  -config configs/mnist_full.config -name MNIST
```
In this case, the program will ask you for an input image.

For any further question please contact to jose.saavedra@orand.cl.
