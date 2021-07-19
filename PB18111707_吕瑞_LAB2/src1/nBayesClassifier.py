import numpy as np
import pandas as pd # 导入 pandas 依赖包
import math
from collections import Counter
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def fit(self,traindata,trainlabel,featuretype):
        '''
        input:  traindata: 训练集数据
                trainlabel: 训练集标签
                featuretype: 对应每个特征的数据类型
        output: self.Pc: 先验概率 p(c)
                self.Pxc: 条件概率分布 p(x|c)
        '''
        data = np.hstack((traindata,trainlabel))
        df = pd.DataFrame(data)
        df = df.rename(columns={8:'label'})
        value_counts = dict(df['label'].value_counts())
        num_of_class=len(value_counts)
        num_of_samples = df[0].count()

        # 计算先验概率
        for i in value_counts:
            pc = (value_counts[i]+1)/(num_of_samples+num_of_class)
            self.Pc[i] = pc

        # 特征列表：feature_list
        feature_list = list(df.columns)
        feature_list.remove('label')

        # 计算条件分布概率
        data_CX = {}
        for i in value_counts:
            data_DivideByLabel= df[df['label']==i]
            data_CX[i] = {}
            for j in feature_list:
                if featuretype[j] == 0:
                    # 离散数据：统计在 i 类别，j 特征下的属性值
                    data_CX[i][j] = dict(data_DivideByLabel[j].value_counts())
                else:
                    # 连续数据：假设数据服从高斯分布，记录平均值，方差，标准差
                    data_CX[i][j] = {
                        'mean':data_DivideByLabel[j].mean(),
                        'var':data_DivideByLabel[j].var(),
                        'std':data_DivideByLabel[j].std()
                    }
            # 统计类别为 i 的数据个数 = D_c
            data_CX[i]['count'] = value_counts[i]
        self.Pxc = data_CX
    



    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        input:  features: 测试集数据特征
                feature_type: 对应特征的数据类型
                self.Pc: 先验概率分布
                self.Pxc: 条件概率分布
        output: np.array(y): 测试结果
        '''
        y = []
        for x in features:
            res_set = {}
            for c in self.Pc:
                res = np.log(self.Pc[c])
                for i in self.Pxc[c]:
                    if i == 'count':
                        continue
                    if featuretype[i] == 0:
                        # 离散数据
                        res = res+np.log((self.Pxc[c][i][x[i]]+1)/(self.Pxc[c]['count']+len(self.Pxc[c][i])))
                    else:
                        #连续数据
                        formula1 = np.exp(-(x[i]-self.Pxc[c][i]['mean'])**2/(2*self.Pxc[c][i]['var']))
                        formula2 = 1/np.sqrt(2*np.pi*self.Pxc[c][i]['std'])
                        res = res+np.log(formula1*formula2)
                res_set[c]=res

            y.append([max(res_set,key=res_set.get)])

        return np.array(y)



def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()