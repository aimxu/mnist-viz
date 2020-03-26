# mnist-viz
Mnist feature viualiztion for each layer

Make sure your environment including:
>tensorflow 1.8.0 (cpu or gpu) <br>
numpy 1.16.4 <br>
matplotlib 3.1.1 <br>
sklearn 0.22.1 <br>

`mnist-train.py` is use for train, it will generate
[weight](https://drive.google.com/open?id=1qznx2T0klYSXXk1G1hzoZhs1T_MhOow8) files in `./model` folder <br>
using LeNet with a little change, the network like following<br>
![image](https://github.com/aimxu/mnist-viz/blob/master/images/network.png)


`mnist-test.py` using weight files to test a random file in mnist.

The datset is download by code, You may have some warning with hign version tensorflow (including this 1.8.0), ,You can also download data first, in LeCun's [website](http://yann.lecun.com/exdb/mnist/index.html).

The visualization of feature like the flowing image<br>
In the code, There are only visualize 2 convolutional layers 1 fully connection layer and output layer.  You can visualize more layers by changing the code.

![image](https://github.com/aimxu/mnist-viz/blob/master/images/viz-feature.png)
