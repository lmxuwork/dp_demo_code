## Tensorflow 实现　YOLOv3－Tiny
- 离散类型的是分类算法，连续类型的是回归算法
- CNN本质的作用是用来分类，定位功能并没有做到，而Yolo这种方法就是只通过CNN网络，就能够实现目标的定位和识别.
- YOLO将物体检测作为回归问题求解。基于一个单独的end-to-end网络，完成从原始图像的输入到物体位置和类别的输出.
- YoLo系列原理
  - YOLO的训练和检测均是在一个单独的网络中进行的.
  - YOLO将物体检测作为一个回归问题进行求解，输入图像经过一次inference（推理），便能得到图像中所有物体的位置和其所属类别及相应的置信概率。
## YoLo的网络结构
  - 