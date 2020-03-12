
#第二天： 数学作业

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
# 引入中文字库
matplotlib.rcParams['font.sans-serif'] = 'SimHei'

#01 读取数据： “homework.npz”
file = np.load("homework.npz")
X = file['X']
d = file['d']
# X = np.random.normal(0, 4, [1000]) 
# d = X ** 2 + X + np.random.normal(0, 0.3, [1000]) 

#02 预处理：数据中是否存在异常数据，并进行排除, 空。

#03 观察：观察给定数据X与d的形式，并进行绘图
# print("X Shape:{}; d Shape:{}".format(np.shape(X), np.shape(d)))
# plt.scatter(X,d)
# plt.show()

#04 通过观察，给定函数y=f(x)的形式
#   假定数据形式为：y=ax+b
#   图像显示数据为连续性分布，所以选择二范数来衡量y与d之间相似程度：loss=(y-d)^2

#05 建模：通过观察，给定函数y=f(x)的形式
def model(x,w):
    """构建二次模型"""
    a,b,c = w
    return func(x**2 *a + x*b + c)
def func(x): #激活函数Relu 
    ret = np.array(x)
    ret[x<0] = 0
    return ret
def dfunc(x): #func（Relu函数）的导数，
    ret = np.zeros_like(x)
    ret[x>0] = 1
    return ret
#06 求解：求解建模过程中的函数参数
# 定义函数关于可训练参数的偏导数
def grad_model(x, d, w):
    a, b, c = w
    y = model(x, w)
    dy = dfunc(x**2 *a + x*b + c)
    grad_a = 2 * (y - d) * dy * x**2
    grad_b = 2 * (y - d) * dy * x
    grad_c = 2 * (y - d) * dy
    return grad_a, grad_b, grad_c
# 6.1 定义初始值
w = [0.001, 0.001, 0.001] # w = [a,b，c]
eta = 0.02  # 学习步长 
batchsize=20 #
# 6.2 一次输出一个样本：计算单个可训练参数的梯度，并进行迭代
def itr_method_one():
    for itr in range(1000):
        idx = np.random.randint(0, len(X))
        inx = X[idx]
        ind = d[idx]
        ga, gb, gc = grad_model(inx, ind, w)
        w[0] -= eta * ga
        w[1] -= eta * gb
        w[2] -= eta * gc
    predict_and_display(w)
# 6.3 一次输入多个样本：计算多个可训练参数的梯度取平均，并进行迭代
def itr_method_two():
    for itr in range(100):
        sum_ga, sum_gb, sum_gc = 0, 0, 0
        for _ in range(batchsize):
            idx = np.random.randint(0, len(X))
            inx = X[idx]
            ind = d[idx]
            ga, gb, gc = grad_model(inx, ind, w)
            sum_ga += ga
            sum_gb += gb
            sum_gc += gc
        w[0] -= eta * sum_ga / batchsize
        w[1] -= eta * sum_gb / batchsize
        w[2] -= eta * sum_gc / batchsize
    predict_and_display(w)
#07 预测某点预期输出值
def predict_and_display(w):
    x = np.linspace(-2, 4, 100)
    y = model(x, w)
    plt.scatter(X[:, 0], d[:, 0],s=20,alpha=0.4, label="数据散点")
    plt.plot(x, y,c="r",alpha=0.5, label="预测关系")
    plt.show()

def main():
    #itr_method_one()
    itr_method_two()

if __name__ == "__main__":
    main()