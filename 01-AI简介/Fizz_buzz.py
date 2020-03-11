import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

## 题目描述
# 面试官让写个程序来玩Fizz Buzz. 这是一个游戏。玩家从1数到100，
# 如果数字被3整除，那么喊'fizz'，如果被5整除就喊'buzz'，
# 如果两个都满足就喊'fizzbuzz'，不然就直接说数字。这个游戏玩起来就像是：
# > 1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 ...
"""传统算法 Rule Based"""
def rule_base():
    res = []
    for i in range(1, 101):
        # 对15取余为0 输出fizzbuzz
        if i % 15 == 0:
            res.append('fizzbuzz')
        # 对3取余为0，输出fizz
        elif i % 3 == 0:
            res.append('fizz')
        # 对5取余为0，输出为buzz
        elif i % 5 == 0:
            res.append('buzz')
        # 不符合以上3种情况，直接输出数字
        else:
            res.append(str(i))
    print(' '.join(res))

"""机器学习的方式"""
def feature_engineer(i):
    """特征工程，"""
    return np.array([i % 3, i % 5, i % 15])

# 将需要预测的指标转换为数字方法：将数据的真实值（预测结果）number, "fizz", "buzz", "fizzbuzz"
# 分别对应转换为数字 3, 2, 1, 0，这样后续能被计算机处理
def construct_sample_label(i):
    if   i % 15 == 0: return np.array([3])
    elif i % 5  == 0: return np.array([2])
    elif i % 3  == 0: return np.array([1])
    else:             return np.array([0])

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
        elif(res[i]==2):
            res_str.append("buzz")
        elif(res[i]==3):
            res_str.append("fizzbuzz")

    # print(res)
    print(res_str)

if __name__ == "__main__":
    main()


