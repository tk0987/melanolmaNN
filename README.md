# melanolmaNN
very nested network

These 'good' results need validation. I'm in fear, that this model is overtrained - despite less than 250000 parameters.

The worse part now:


1.	Introduction

This convolutional neural network is called by me „Światowid” because of its architecture – input data feeds four partially independent (but interconnected) computational blocks, which at some point merge into common dense core.
This network is just VGG16 but with multiplication and interconnecting convolutional layers.
Main purpose of this network was image classification.
The output should be a vector, indicating the probability whether an image belongs to class.

Data preparation

Everything is in provided code – User needs a folder tree with images, due to used function: 
tf.keras.preprocessing.image_dataset_from_directory()
Images should be (as for today) equal in size and format.
Dataset was in here:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

First run

It is advised to run the code in debug mode until all problems are solved. User do not need any special software to run this code – ordinary Python console is enough. Please, sort the images first – use sorter.py, described later.

2.	Input shapes

User can change input shapes in the tensorflow.keras.layers.Input layer. It is 600x450.

3.	Hardware limitations

In general, this line in a code is crucial:

130   model = światowid(1,7,32)

„Światowid” function has three arguments: basic number of trainable parameters, number of classes and batch size. The example above results in consuming <8 GB of my GPU VRAM (230000 trainable parameters, Gigabyte RTX 2080).
It is advised to start with model = światowid(1,7,1). 

If User want to run the code on CPU only: please comment the lines connected with keyword „GPU”. Indent properly if needed.

If User want to run the code on Windows WSL2 with GPU, please enter this lines into Linux console before:

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

4.	Smaller codes in project

There is only one (not crucial to run main network):
  1.	sorter.py
This code needs proper folder tree already created. It just checks if the image name belongs to a class, than it copies this image to desired folder in a tree.
This code needs labeled image names – author created a file labels.txt using Microsoft Excel and simple notepad – downloaded dataset had somewhat difficult to understand label assignment.

Please look at the paths in all provided codes – there is high risk of overlooking them.
