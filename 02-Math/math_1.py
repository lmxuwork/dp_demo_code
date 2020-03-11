
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

#02 预处理：数据中是否存在异常数据，并进行排除, 空。

#03 观察：观察给定数据X与d的形式，并进行绘图
# print("X Shape:{}; d Shape:{}".format(np.shape(X), np.shape(d)))
# plt.scatter(X,d)
# plt.show()

#04 通过观察，给定函数y=f(x)的形式
#   假定数据形式为：y=ax+b
#   图像显示数据为连续性分布，所以选择二范数来衡量y与d之间相似程度：loss=(y-d)^2

#05 建模：通过观察，给定函数y=f(x)的形式
def f(x, w): #线性函数
    a, b = w
    return func(a * x + b) #定义函数
def func(x):
    ret = np.array(x)
    ret[x<0] = 0
    return ret
def dfunc(x):
    print("x=",x)
    ret = np.zeros_like(x)
    ret[x>0] = 1
    return ret
#06 求解：求解建模过程中的函数参数
def grad_f(x, d, w):
    a, b = w
    y = f(x, w)
    dy = dfunc(a * x + b)
    grad_a = 2 * (y - d) * dy * x
    grad_b = 2 * (y - d) * dy
    return grad_a, grad_b

# 6.1 定义初始值
w = [0.1,0.1] # w = [a,b],注意不能初始化为0
eta = 0.001 # 学习步长，注意步长不能过大
batchsize=10 #
# 6.2 一次输出一个样本：计算单个可训练参数的梯度，并进行迭代
def itr_method_one():
    eta = 0.1
    for itr in range(1000):
        idx = np.random.randint(0, len(X))
        inx = X[idx]
        ind = d[idx]
        ga, gb = grad_f(inx, ind, w)
        w[0] -= eta * ga
        w[1] -= eta * gb
    predict_and_display(w)
# 6.3 一次输入多个样本：计算多个可训练参数的梯度取平均，并进行迭代
def itr_method_two():
    eta = 0.01
    batchsize=10
    for itr in range(1000):
        sum_ga, sum_gb = 0, 0
        for _ in range(batchsize):
            idx = np.random.randint(0, len(X))
            inx = X[idx]
            ind = d[idx]
            ga, gb = grad_f(inx, ind, w)
            sum_ga += ga
            sum_gb += gb
        w[0] -= eta * sum_ga / batchsize
        w[1] -= eta * sum_gb / batchsize
    predict_and_display(w)

#07 预测某点预期输出值
def predict_and_display(w):
    x = np.linspace(-2, 4, 100)
    y = f(x, w)
    plt.scatter(X[:, 0], d[:, 0],s=20,alpha=0.4, label="数据散点")
    plt.plot(x, y,c="r",alpha=0.5, label="预测关系")
    plt.show()

def main():
    itr_method_one()
    #itr_method_two()

if __name__ == "__main__":
    main()