## Self-taught deep learning, train models and understand the underlying mathematics

- - -
## 笔记介绍

主要笔记是结合吴恩达老师cs229课程、李沐老师的动手学深度学习，deep learning深度学习（花书）自己总结记录的。（日更····）

理论笔记中语言是**Python**，使用的框架是**pytorch**。
工程笔记是智能车百度完全模型，语言是**python**，使用的框架是**paddle**
- [x]  opencv笔记涉及到的理论和代码非常少

理论笔记是针对的深度学习课程，指的是数学公式的推导，背后原理的解释
工程笔记是深度学习数据增强、超参数调节、剪枝等操作代码实际笔记

深度学习的基础：学会构建神经网络，并用在创建自己的机器学习项目。对卷积神经网络 (**CNN**)、递归神经网络 (**RNN**)、长短期记忆 (**LSTM**) 等深度学习常用的网络结构、工具和知识都有涉及。

本人水平有限，如有公式、算法错误，请及时指出

- - -

在探索数学原理与工程实践的两条道路上，我发现它们各自独行其道。在此，笔者提醒勿让自己深陷于数学公式的迷雾之中，而忽略了将这些理论转化为实际代码的艺术。盲目追求理论完美，很容易落入一个甜美的错觉，非常容易自我感动式的付出——自认为洞悉了所有代码的奥秘，却在实际应用时束手无策。

这也是那些习惯于题海战术的同学们常犯的错误，他们在数学的迷宫中迷失方向，耗费了太多宝贵时间。因此，大家不妨先从整体上理解概念，然后立刻着手实践，通过编写代码、上手工程来揭示和掌握背后的原理。当在具体实践中遇到困难时，再回头去寻找理论知识的支持，这样的方法不仅加深了对模型架构的理解，也会在解决问题后有正反馈的兴趣。

待到对基础架构有了一定的了解，再适时回顾那些数学公式的推导，你会发现自己的理解更加精确、全面。这种学习方式，既保证了知识的实际应用，也提升了理论的掌握深度，让每一步都更为恰当和充实。

- - -
## 学习资源

- 视频课程：[**stanford cs229  -  andrew ng**](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

- 指导代码书 : [**d2l-zh**](https://zh-v1.d2l.ai/)

- 深入了解深度学习背后的数学原理：**PRML**

- - -
## 笔者相关书籍推荐 reference book


欢迎各位斧正错误，有什么书籍推荐可以发送至我互相讨论学习

- 入门读物 [The Elements of Statistical Learning(英文第二版),The Elements of Statistical Learning.pdf]

- PRML

- [统计学习方法], (@Dr. Hang Li/李航博士)


- python读物:
    
    - 利用Python进行数据分析
        
    - Python学习手册
        
    - Python性能分析与优化
        
    - Python科学计算(第2版)
        
    - Python计算机视觉编程 [美] Jan Erik Solem
        
    - python核心编程(第三版)
        
    - Python核心编程（第二版）
        
    - Python高手之路 - [法] 朱利安·丹乔（Julien Danjou）
        
    - Python3 CookBook中文版
        
    - 终极算法机器学习和人工智能如何重塑世界 - [美 ]佩德罗·多明戈斯
        
    - 机器学习系统设计 (图灵程序设计丛书) - [美]Willi Richert & Luis Pedro Coelho
        
    - 机器学习实践指南：案例应用解析（第2版） (大数据技术丛书) - 麦好
        
    - 机器学习实践 测试驱动的开发方法 (图灵程序设计丛书) - [美] 柯克（Matthew Kirk）
        
    - 机器学习：实用案例解析

- 数学:
    
    - Algebra - Michael Artin
        
    - Algebra - Serge Lang
        
    - Basic Topology - M.A. Armstrong
        
    - Convex Optimization by Stephen Boyd & Lieven Vandenberghe
        
    - Functional Analysis by Walter Rudin
        
    - Functional Analysis, Sobolev Spaces and Partial Differential Equations by Haim Brezis
        
    - Graph Theory - J.A. Bondy, U.S.R. Murty
        
    - Graph Theory - Reinhard Diestel
        
    - Inside Interesting Integrals - Pual J. Nahin
        
    - Linear Algebra and Its Applications - Gilbert Strang
        
    - Linear and Nonlinear Functional Analysis with Applications - Philippe G. Ciarlet
        
    - Mathematical Analysis I - Vladimir A. Zorich
        
    - Mathematical Analysis II - Vladimir A. Zorich
        
    - Mathematics for Computer Science - Eric Lehman, F Thomson Leighton, Alber R Meyer
        
    - Matrix Cookbook, The - Kaare Brandt Petersen, Michael Syskind Pedersen
        
    - Measures, Integrals and Martingales - René L. Schilling
        
    - Principles of Mathematical Analysis - Walter Rudin
        
    - Probabilistic Graphical Models: Principles and Techniques - Daphne Koller, Nir Friedman
        
    - Probability: Theory and Examples - Rick Durrett
        
    - Real and Complex Analysis - Walter Rudin
        
    - Thomas' Calculus - George B. Thomas
        
    - 普林斯顿微积分读本 - Adrian Banner
    
    - Learning Data Mining with Python
        
    - Matplotlib for python developers
        
    - Machine Learing with Spark
        
    - Mastering matplotlib
        
    - Neural Network Programming with Java
        
    - Python Machine Learning
        
    - Java Deep Learning Essentials
        
    - Learning Pandas
        
    - Python Parallel Programming Cookbook


- - -
## 所需要的配置(ubuntu22.04)

#### [**anaconda**](https://www.anaconda.com/)
Check installation
```
conda --version
```

Add system path
```
export PATH="/path/to/anaconda3/bin:$PATH"
```

Initialize Anaconda
```
source ~/anaconda3/bin/activate
conda init
```

Create an Anaconda environment
```
conda create -n xxx python=3.x(Change these values as needed with x)
```


Activate the Anaconda environment
```
conda activate xxx
```

#### **install  CUDA (12.0)**
```
# install CUDA 12.0

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget http://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu1804-12-0-local_12.0.0-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1804-12-0-local_12.0.0-1_amd64.deb

sudo apt-key add /var/cuda-repo-ubuntu1804-12-0-local/7fa2af80.pub

sudo apt-get update

sudo apt-get install cuda

```

#### **install PyTorch（GPU 11.0**)
You need to choose a version that is adapted to CUDA 11.0, because PyTorch is usually distinguished by CUDA version.

```
# conda install PyTorch

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

# or use pip

pip install torch torchvision torchaudio
```

#### **install MXNet（cu112 version）**
```
pip install mxnet-cu112
```

#### **install Keras (2.6)**
```
pip install keras==2.6
```

#### **install opencv**
OpenCV is a computer vision library that is open-source and free for use. It is widely used for a variety of applications, such as facial recognition, object tracking, and motion detection. OpenCV was developed by Intel and is written in C++, with interfaces available for Python, Java, and other programming languages.
```
sudo apt install libopencv-dev python3-opencv
```

The most current version of the OpenCV library may be obtained by compiling it from its source code. You’ll have full say over how the build is tailored to your machine’s specifications. This method is recommended for setting up OpenCV. To install the most recent OpenCV version directly from the source, follow these instructions:

Make backups of OpenCV and all its source code repositories：
```
mkdir ~/opencv_build && cd ~/opencv_build

git clone https://github.com/opencv/opencv.git

git clone https://github.com/opencv/opencv_contrib.git
```

Make a temporary directory to construct in, and then go to it after the download is complete:
```
cd ~/opencv_build/opencv

mkdir -p build && cd build
```

To make OpenCV, only need to setup CMake:
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \

    -D CMAKE_INSTALL_PINEFIX=/usr/local/ \

    -D INSTALL_C_EXAMPLES=ON \

    -D INSTALL_PYTHON_EXAMPLES=ON \

    -D OPENCV_GENERATE_PKGCONFIG=ON \

    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \

    -D BUILD_EXAMPLES=ON ..

```

  
need to start the compilation process using:
```
make -j8
```

Modify the -j option to suit your CPU. The time required to create is system-specific, so if you’re not sure how many cores your CPU has, run nproc into the terminal to find out.

The next step is to Set up the OpenCV you can do this using the below command:
```
sudo make install
```

 Type the following steps to see the OpenCV version to ensure proper installation.

```
 pkg-config --modversion opencv4
 python3 -c "import cv2; print(cv2.__version__)"
```