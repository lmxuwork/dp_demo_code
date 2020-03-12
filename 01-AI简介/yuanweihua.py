import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

"""
* 问题1 通过KNN进行鸢尾花分类任务
* 问题2 参数搜索：搜索最优的K近邻，使得模型准确度最高，观察准确度最高的K取值
"""

def get_iris_data():
    iris = load_iris() #加载并返回鸢尾花数据集
    iris_data = iris.data # 鸢尾花特征值(4个)
    iris_target = iris.target # 鸢尾花目标值(类别)
    return iris_data, iris_target

#问题一
def func_one(x_train, x_test, y_train, y_test):
    """ 
    03 特征工程(对特征值进行标准化处理)
        由于每个特征的大小，取值范围等不一样，这样会导致每个特征的权重不一样，而实际上是一样的。
        通过对原始数据进行变换把数据变换到均值为0,方差为1范围内。这样每个特征值的权重变得一样，以便于计算机处理。
    """
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    """
    04 训练
    """
    knn = KNeighborsClassifier(n_neighbors=5) # 创建一个KNN算法实例，n_neighbors默认为5
    knn.fit(x_train, y_train) # 将测试集送入算法
    y_predict = knn.predict(x_test) # 获取预测结果

    """
    05 预测结果展示
    """
    labels = ["山鸢尾","虹膜锦葵","变色鸢尾"]
    for i in range(len(y_predict)):
        print("第%d次测试:真实值:%s\t预测值:%s"%((i+1),labels[y_predict[i]],labels[y_test[i]]))
    print("准确率：",knn.score(x_test, y_test))

"""
问题2 参数搜索：搜索最优的K近邻，使得模型准确度最高，观察准确度最高的K取值
"""
def func_two(x_train, x_test, y_train, y_test):
    # 生成knn估计器
    knn = KNeighborsClassifier()
    # 构造超参
    params = {"n_neighbors":[3,5,10]}
    #knn：估计器对象 params：估计器参数(dict){“n_neighbors”:[1,3,5]} cv：指定几折交叉验证
    gridCv = GridSearchCV(knn, param_grid=params,cv=5)
    gridCv.fit(x_train,y_train) # 输入训练数据
    # 预测准确率
    print("参数搜索 准确率：",gridCv.score(x_test, y_test))
    print("交叉验证中最好的结果：",gridCv.best_score_)
    print("最好的模型：", gridCv.best_estimator_)

def main():
    """ 
    01 读取数据 
    """
    iris_data, iris_target = get_iris_data() 
    #print(iris_data, iris_target)
    """ 
    02 划分数据集:
        x_train 训练集特征值 x_test 测试集特征值 y_train 训练集目标值 y_test 测试集目标值 test_size=0.25 表示25%的数据用于测试
    """
    x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.25)

    func_one(x_train, x_test, y_train, y_test)
    func_two(x_train, x_test, y_train, y_test)
   
    """
    结果
       n_neighbors=10的时候，该模型的效果最好
    """
if __name__ == "__main__":
    main()