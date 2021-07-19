'''
Author: karin
Date: 2021-07-09 13:35:54
LastEditors: karin
LastEditTime: 2021-07-14 15:19:27
Description: file content
TimeComplexity: O()
FilePath: \AILAB\LAB2_for_students\src1\SVM.py
'''
import numpy as np
import cvxopt #用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc


#根据指定类别main_class生成1/-1标签
def svm_label(labels,main_class):
    new_label=[]
    for i in range(len(labels)):
        if labels[i]==main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)

# 实现线性回归
class SupportVectorMachine:

    '''参数初始化
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,kernel,C,Epsilon):
        self.kernel=kernel
        self.C = C
        self.Epsilon=Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''
    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        #d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1,x2)
        elif kernel == 'Poly':
            K = np.dot(x1,x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''
    def fit(self,train_data,train_label,test_data):
        '''
        input:
                - train_data: 训练集特征
                - train_lable: 训练集标签
                - test_data: 测试集特征

        output:
                - Y: 测试集预测结果
        '''
        m = len(train_label)
        P = np.zeros((m, m))
        for i in range(0, m):
            for j in range(0, m):
                P[i][j] = train_label[i]*train_label[j] * \
                    self.KERNEL(train_data[i], train_data[j], self.kernel)
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(-np.ones((m, 1)))
        A = np.zeros((m, 1))
        for i in range(0, m):
            A[i] = train_label[i]
        A = A.T
        A = cvxopt.matrix(A)
        b = cvxopt.matrix([0.0])
        G1 = -np.eye(m)
        G2 = np.eye(m)
        G = cvxopt.matrix(np.vstack((G1, G2)))
        h1 = np.zeros((m, 1))
        h2 = np.ones((m, 1))*self.C
        h = cvxopt.matrix(np.vstack((h1, h2)))

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b) # 求解标准二次规划问题
        X = np.array(sol['x']) # 得出 a 的结果
        X_NZ = []
        # 根据阈值设置 a 的值
        for i in range(0,m):
            if X[i] < self.Epsilon:
                X[i] = 0
            else:
                X_NZ.append(i) # 记录非零下标值

        len_XNZ=len(X_NZ)
        b_test = []
        for i in X_NZ:
            temp = 0
            for j in X_NZ:
                temp += self.KERNEL(train_data[i], train_data[j],
                            self.kernel)*X[j]*train_label[j]
            b_test.append([train_label[i]-temp])

        b_test = np.array(b_test)
        b_final =np.mean(b_test) # 取均值作为 b

        # 测试结果
        test_len = len(test_data)
        y = np.zeros((test_len,1))
        for i in range(0,test_len):
            temp = 0
            x_i = test_data[i] # 待测试样本
            for j  in range(0,m):
                # 训练集维度的循环
                temp +=X[j]*train_label[j]*self.KERNEL(x_i,train_data[j],self.kernel)
            y[i] = temp + b_final
        return y




def main():
    # 加载训练集和测试集
    Train_data,Train_label,Test_data,Test_label=load_and_process_data()
    Train_label=[label[0] for label in Train_label]
    Test_label=[label[0] for label in Test_label]
    train_data=np.array(Train_data)
    test_data=np.array(Test_data)
    test_label=np.array(Test_label).reshape(-1,1)
    #类别个数
    num_class=len(set(Train_label))


    #kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    #C为软间隔参数；
    #Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    kernel='Gauss'
    C = 1
    Epsilon=10e-5
    #生成SVM分类器
    SVM=SupportVectorMachine(kernel,C,Epsilon)

    predictions = []
    #one-vs-all方法训练num_class个二分类器
    for k in range(1,num_class+1):
        #将第k类样本label置为1，其余类别置为-1
        train_label=svm_label(Train_label,k)
        # 训练模型，并得到测试集上的预测结果
        prediction=SVM.fit(train_data,train_label,test_data)
        predictions.append(prediction)
    predictions=np.array(predictions)
    #one-vs-all, 最终分类结果选择最大score对应的类别
    pred=np.argmax(predictions,axis=0)+1

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
