# Multiresolution Tree Networks for 3D Point Cloud Processing (ECCV 2018)


This repository contains the source code for the ECCV 2018 paper Multiresolution Tree Networks for 3D Point Clout Processing. 

[Project page](http://mgadelha.me/mrt/)

![MRTNet reconstructions](http://mgadelha.me/mrt/fig/realrec2.png)

## Dependencies

* numpy
* pytorch
* tensorboardX
* fxia22/pointGAN (optional - if you want faster Chamfer Distance) Thanks to Fei Xia'a for making the code publicly available. We have a version in this repo already, but might not be up-to-date.


## Train

First, you need to change the dataset path in the file `train_img2pc.py`. We are using the rendered images from https://github.com/chrischoy/3D-R2N2. You will also need a path for the point clouds in .npy format. Finally, you can train a model by using the following command:
```
python train_img2pc.py --name experiment_name
```

If you want to run the model, change the folder name indicated in `run_img2pc.py` and use the following command:
```
python run_img2pc.py --n experiment_name
```
Notice that`experiment_name` should match in both cases. Similarly, we also have evaluation code to reproduce the paper's numbers.


## Point Sampling

This repository also contains code for sampling the point clouds in the sampler folder. It is a single .java file and it contains a README with specific instructions.
This code automatically sorts the points according to a kd-tree structure.


## Citation

If you use any part of this code or data, consider citing this work:
```
@inProceedings{mrt18,
  title={Multiresolution Tree Networks for 3D Point Cloud Processing},
  author = {Matheus Gadelha and Rui Wang and Subhransu Maji},
  booktitle={ECCV},
  year={2018}
}
```
