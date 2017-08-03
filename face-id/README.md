# face identification demo

This demo runs face identification based on [LightCNN](https://github.com/AlfredXiangWu/LightCNN/)


### Initial setup

 
##### Convert model:
 
In order to convert the model from LightCNN for CPU use, we run:

`convert-cpu-model.py`

##### Collect dataset data from camera:

`python3 demo-face.py --mode 2`
 
This collects a small amount of samples and creates a database for demo to work on.

##### Train database from a folder of images:

This will extract a dataset from a folder of folders (name/id) of face images:

`python3 demo-face.py --mode 3 --face_images_dir /your/data/dir/ `

 
### Demo
 
Run demo (mode 1) as:

`python3 demo-face.py`

Before using, you need to create your database of users examples. This is done by standing in front of camera and running: