import random

from scipy.io import loadmat
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from argparse import ArgumentParser

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def visualize_eig_values(eig_values):
    plt.figure(figsize=(5, 3))
    limit = 100 if len(eig_values) > 100 else len(eig_values)
    plt.stem(eig_values[:limit])
    plt.xlabel('Eigen value index')
    plt.ylabel('Eigen value')
    plt.show()


def PCA(X, num_components, normalized=True, visualize=False):
    """
    主成分分析
    :param X:  (num_examples,dims)
    :param num_components: number of principal components
    :param normalized: whether to scale the input data
    :return: (num_examples,num_components)
    """
    if normalized:
        mean_ = X.mean(axis=0)
        std_ = X.std(axis=0)
        X_scaled = (X - mean_) / std_
    else:
        X_scaled = X

    cov_mat = np.cov(X_scaled.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    if visualize:
        visualize_eig_values(sorted_eigenvalue)
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
    X_reduced = np.dot(eigenvector_subset.transpose(), X_scaled.transpose()).transpose()
    return X_reduced


class KNN():
    def __init__(self, k=1):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        final_labels = []
        for sample in x_test:
            dist = np.linalg.norm(sample - self.x_train, axis=1)
            nearest_neighbor_ids = dist.argsort()[:self.k]
            labels = self.y_train[nearest_neighbor_ids]
            counts = np.bincount(labels)
            sample_label = np.argmax(counts)
            final_labels.append(sample_label)
        return final_labels


class lda(object):
    def __init__(self, data, label, n_dim):
        """
        初始化lda
        :param data: 要降维的数据
        :param label: 数据标签
        :param n_dim: 保留的维度
        """
        self.X = data
        self.y = label
        self.n_dim = n_dim
        self.clusters = np.unique(label)
        self.lda_data = None
        # assert n_dim == len(self.clusters), "your target dimension is too big!"
        assert len(self.X) == len(self.y), "the length of label must be equal to your data"

    def zero_mean(self, x):
        """
        零均值化特征矩阵m*n&#xff0c;m为样本个数&#xff0c;n为特征维度
        :return: zero_mean 0均值矩阵
        """
        average = np.mean(x, axis=0)  # 对数组第0个维度求均值&#xff0c;就是求每列的均值 得到每个特征的平均 1*n
        zero_mean = x - average
        return zero_mean

    def get_Sw(self):  # 求类内散度矩阵
        Sw = np.zeros((self.X.shape[1], self.X.shape[1]))  # 初始化散度矩阵
        for i in self.clusters:  # 对每个类别分别求类内散度后相加
            data_i = self.X[self.y == i]
            Swi = np.mat(self.zero_mean(data_i)).T * np.mat(self.zero_mean(data_i))
            Sw += Swi
        return Sw

    def get_Sb(self):  # 求类间散度矩阵&#xff0c;即全局散度-类内散度
        temp = np.mat(self.zero_mean(self.X))
        St = temp.T * temp
        Sb = St - self.get_Sw()
        return Sb

    def start_lda(self, visualize=True):
        """
        开始lda降维, 并记录保留的特征占比
        :return:
        data_ndim&#xff1a;降维后的数据
        """
        Sb = self.get_Sb()
        Sw = self.get_Sw()
        S = np.linalg.inv(Sw) * Sb  # 计算矩阵Sw^-1*Sb
        eigVals, eigVects = np.linalg.eig(S)  # 求S的特征值&#xff0c;特征向量
        index = np.argsort(-eigVals)  # 按照特征值进行从大到小排序

        # 计算保留的特征占比
        # eigVals_ = eigVals[index]  # 得到重新排序后的特征值
        # eigVals_ = np.asarray(eigVals_).astype(float)
        # featSum = np.sum(eigVals_)
        # featSum_ = 0.0
        # for i in range(self.n_dim):
        #     featSum_ = featSum_ &#43; eigVals_[i]
        # proportion = featSum_ / featSum * 100
        # print('the proportion of remained feature is:', proportion)

        w = np.mat(eigVects[:, index[:self.n_dim]])  # 选出前n_dim个特征向量&#xff0c;保存为矩阵
        data_ndim = np.asarray(self.X * w).astype(float)
        self.lda_data = data_ndim
        if visualize:
            self.plot_2D_feat()
        return data_ndim

    def plot_2D_feat(self):
        """
        绘制前两个维度的图像&#xff0c;并和sklearn中的pca方法进行对比
        """
        assert self.lda_data is not None, "please start pca before plot"  # 抛出异常

        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title("my LDA")
        plt.scatter(self.lda_data[:, 0], self.lda_data[:, 1], c=self.y)
        plt.xlabel('x1')
        plt.ylabel('x2')

        # 绘制sklearn的lda图像
        skl_lda = LinearDiscriminantAnalysis(n_components=self.n_dim).fit_transform(self.X, self.y)
        plt.subplot(122)
        plt.title("sklearn_LDA")
        plt.scatter(skl_lda[:, 0], skl_lda[:, 1], c=self.y)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()
        plt.savefig("LDA.png", dpi=600)
        return 0


def LDA(data, target, n_dim, normalized=True):
    '''
    线性判别分析
    :param data: (n_samples, n_features)
    :param target: data class
    :param n_dim: target dimension
    :return: (n_samples, n_dims)
    '''
    if normalized:
        mean_ = data.mean(axis=0)
        std_ = data.std(axis=0)
        X_scaled = (data - mean_) / std_
    else:
        X_scaled = data
    data = X_scaled

    clusters = np.unique(target)

    if n_dim > len(clusters) - 1:
        print("K is too much")
        print("please input again")
        exit(0)

    # within_class scatter matrix
    Sw = np.zeros((data.shape[1], data.shape[1]))
    for i in clusters:
        datai = data[target == i]
        datai = datai - datai.mean(0)
        Swi = np.mat(datai).T * np.mat(datai)
        Sw += Swi

    # between_class scatter matrix
    SB = np.zeros((data.shape[1], data.shape[1]))
    u = data.mean(0)  # 所有样本的平均值
    for i in clusters:
        Ni = data[target == i].shape[0]
        ui = data[target == i].mean(0)  # 某个类别的平均值
        SBi = Ni * np.mat(ui - u).T * np.mat(ui - u)
        SB += SBi
    S = np.linalg.inv(Sw) * SB
    eigVals, eigVects = np.linalg.eig(S)  # 求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim - 1):-1]
    w = eigVects[:, eigValInd]
    data_ndim = np.dot(data, w)

    return data_ndim


def load_orl(file="ORLData_25.mat"):
    mat = loadmat(file)
    X = mat["ORLData"].T  # (400,645)
    np.random.shuffle(X)
    Y = X[:, -1]  # (400,)
    X = X[:, :-1]  # (400,644)
    return X, Y


def train_test_split(X, Y, split_ratio=0.8):
    n, d = X.shape
    # split_ratio = 0.8
    split = int(split_ratio * n)
    x_train = X[:split, :]
    y_train = Y[:split]
    x_test = X[split:, :]
    y_test = Y[split:]
    return x_train, y_train, x_test, y_test


def load_vehicle(file="./vehicle.mat"):
    mat = loadmat(file)
    X = mat['UCI_entropy_data'][0, 0]['train_data'].T  # (846,19)
    np.random.shuffle(X)
    Y = X[:, -1]  # (400,)
    X = X[:, :-1]  # (400,644)
    return X, Y


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orl_dim", type=int, default=15)
    parser.add_argument("--vehicle_dim", type=int, default=3)
    parser.add_argument("--K", type=int, default=1)
    args = parser.parse_args()
    split_ratio = args.split_ratio
    set_seed(args.seed)

    '''
    ORL data
    '''

    print("Loading orl dataset...")
    X_ori, Y_ori = load_orl()
    dim = args.orl_dim
    # ==================== LDA + KNN ====================//
    X, Y = X_ori, Y_ori
    # X = LDA(X,Y,n_dim=30)
    # X = LinearDiscriminantAnalysis(n_components=15).fit_transform(X,Y)

    lda_1 = lda(X, Y, n_dim=dim)
    X = lda_1.start_lda(visualize=True)

    x_train, y_train, x_test, y_test = train_test_split(X, Y, split_ratio=split_ratio)

    # knn
    classifier = KNN(k=args.K)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = np.sum(y_pred == y_test) / x_test.shape[0]
    print("| dim: {} | LDA+KNN: {} |".format(dim, acc))

    # ================= PCA + KNN========================//
    X, Y = X_ori, Y_ori
    X = PCA(X, dim, visualize=True)
    x_train, y_train, x_test, y_test = train_test_split(X, Y, split_ratio=split_ratio)

    # knn
    classifier = KNN(k=args.K)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = np.sum(y_pred == y_test) / x_test.shape[0]
    print("| dim: {} | PCA+KNN: {} |".format(dim, acc))

    '''
    Vehicle data
    '''

    print("Loading vehicle dataset...")
    X_ori, Y_ori = load_vehicle()
    dim_1 = args.vehicle_dim

    # =================== LDA + KNN ================//
    X, Y = X_ori, Y_ori
    # X = LDA(X,Y,n_dim=30)
    # X = LinearDiscriminantAnalysis(n_components=15).fit_transform(X,Y)
    lda_2 = lda(X, Y, n_dim=dim_1)
    X = lda_2.start_lda(visualize=True)

    x_train, y_train, x_test, y_test = train_test_split(X, Y, split_ratio=split_ratio)
    # knn
    classifier = KNN(k=args.K)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = np.sum(y_pred == y_test) / x_test.shape[0]
    print("| dim: {} | LDA+KNN: {} |".format(dim_1, acc))

    # *********** PCA + KNN *********************
    X, Y = X_ori, Y_ori
    X = PCA(X, dim_1, visualize=True)
    x_train, y_train, x_test, y_test = train_test_split(X, Y, split_ratio=split_ratio)

    # knn
    classifier = KNN(k=args.K)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = np.sum(y_pred == y_test) / x_test.shape[0]
    print("| dim: {} | PCA+KNN: {} |".format(dim_1, acc))
