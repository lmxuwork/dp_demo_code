
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

# 【选做】：将问题改变，然后参考实例实现基于规则和基于机器学习的方式进行解决。
# 新问题：面试官让写个程序来玩Fizz Buzz. 这是一个游戏。
# 玩家从1数到100，如果数字被3整除，那么喊'fizz'，如果不被3整除就直接说数字。这个游戏玩起来就像是
# > 1 2 fizz 4 5 fizz 7 8 ...
"""传统算法 Rule Based"""
def rule_base():
    res = []
    for i in range(1, 101):
        if i % 3 == 0:
            res.append('fizz')
        else:
            res.append(str(i))
    print(' '.join(res))

def feature_engineer(i):
    return np.array([i % 3])

def construct_sample_label(i):
    if i % 3  == 0: return np.array([1])
    else:           return np.array([0])

def create_datasets():
    #训练集
    x_train = np.array([feature_engineer(i) for i in range(101, 200)])
    y_train = np.array([construct_sample_label(i) for i in range(101, 200)])
    #测试集
    x_test = np.array([feature_engineer(i) for i in range(1, 100)])
    y_test = np.array([construct_sample_label(i) for i in range(1, 100)])
    return x_train, y_train,x_test,y_test

def main():
    print("=======Rule Base=========")
    rule_base()
    print("===========AI============")
    x_train, y_train,x_test,y_test = create_datasets() #01 生成数据 训练集与测试集
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train.ravel())
    print('knn train score: %f'
       % knn.score(x_train, y_train))
    print('knn test score: %f'
        % knn.score(x_test, y_test))
    res=knn.predict(x_test[0:100])
    res_str = []
    for i in range(len(res)):
        if(res[i]==0):
            res_str.append(i+1)
        elif(res[i]==1):
            res_str.append("fizz")
    # print(res)
    print(res_str)

if __name__ == "__main__":
    main()