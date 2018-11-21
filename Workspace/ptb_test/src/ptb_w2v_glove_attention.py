# coding: utf-8
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import numpy as np
import collections
import pickle as pk
#import os

#os.environ['CUDA_VISIBLE_DEVICES']='0'#指定使用第一块GPU
config=tf.ConfigProto()
config.gpu_options.allow_growth=True#增长式
#config.gpu_options.per_process_gpu_memory_fraction=0.9#百分比模式

home_dir='/home/liu/ptb_test'
data_path=home_dir+'/corpus'
vectorfile='vector'
#注意是否为二进制，后面代码要相应修改
vectorname='glove.txt'

#预处理部分
#构造词典，将corpus转化为索引序列
def build_dataset():
    train_path = data_path+"/ptb.train.txt"
    valid_path = data_path+"/ptb.valid.txt"
    test_path = data_path+"/ptb.test.txt"
    
    fin=open(train_path,'r',encoding='utf-8')
    line=fin.read()
    line_f=line.split(' ')
    print('train读取语料完毕')
    #释放空间
    del line
    while ''in line_f:
        line_f.remove('')
    fin.close()
    
    count=[['OOV',0]]
    #前面是word后面是频率
    count.extend(collections.Counter(line_f).most_common())
    dictionary=dict()
    for word,_ in count:
        if _ >=0:
            dictionary[word]=len(dictionary)
    #train        
    train_data=list()
    unk_count=0
    for word in line_f:
        if word in dictionary:
            index=dictionary[word]
            train_data.append(index)
        else:
            unk_count+=1
    print('Train_data Unknown words: %d \nDictionary length: %d \nCorpus length: %d \n' % (unk_count,len(dictionary),len(train_data)))   
    
    #valid
    fin=open(valid_path,'r',encoding='utf-8')
    line=fin.read()
    line_f=line.split(' ')
    print('valid读取语料完毕')
    #释放空间
    del line
    while ''in line_f:
        line_f.remove('')
    fin.close()
    
    valid_data=list()
    unk_count=0
    for word in line_f:
        if word in dictionary:
            index=dictionary[word]
            valid_data.append(index)
        else:
            unk_count+=1
    print('Valid_data Unknown words: %d \nDictionary length: %d \nCorpus length: %d \n' % (unk_count,len(dictionary),len(valid_data)))  
    
    #test
    fin=open(test_path,'r',encoding='utf-8')
    line=fin.read()
    line_f=line.split(' ')
    print('valid读取语料完毕')
    #释放空间
    del line
    while ''in line_f:
        line_f.remove('')
    fin.close()
    
    test_data=list()
    unk_count=0
    for word in line_f:
        if word in dictionary:
            index=dictionary[word]
            test_data.append(index)
        else:
            unk_count+=1
    print('Test_data Unknown words: %d \nDictionary length: %d \nCorpus length: %d \n' % (unk_count,len(dictionary),len(test_data))) 
    
    dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    
    #将corpus，dictionary保存为pickle
    f1=open(data_path+'/train_data','wb')
    pk.dump(train_data,f1,2)
    f1.close()
    f1=open(data_path+'/valid_data','wb')
    pk.dump(valid_data,f1,2)
    f1.close()
    f1=open(data_path+'/test_data','wb')
    pk.dump(test_data,f1,2)
    f1.close()
    f2=open(data_path+'/dict_en','wb')
    pk.dump(dictionary,f2,2)
    f2.close()
    print('pickle保存完毕')
    return train_data,valid_data,test_data,dictionary

def load_data():
    #读取pickle文件
    f1=open(data_path+'/train_data','rb')
    train_data=pk.load(f1)
    f1.close()
    f1=open(data_path+'/valid_data','rb')
    valid_data=pk.load(f1)
    f1.close()
    f1=open(data_path+'/test_data','rb')
    test_data=pk.load(f1)
    f1.close()
    f2=open(data_path+'/dict_en','rb')
    dictionary=pk.load(f2)
    f2.close()
    print('pickle读取完毕')
    return train_data,valid_data,test_data,dictionary

#train_data,valid_data,test_data,dictionary=build_dataset()
train_data,valid_data,test_data,dictionary=load_data()
    
#初始化词向量位置p矩阵
w1 = KeyedVectors.load_word2vec_format(home_dir+'/%s/%s'%(vectorfile,vectorname), encoding='utf-8',binary=False)
print('初始化位置词向量矩阵p完毕')

#全局变量设置
vocab_size=len(dictionary)
window=20
embedding_size=300
batch_size=24
num_steps_train=len(train_data)//window//batch_size
num_steps_valid=len(valid_data)//window//batch_size
num_steps_test=len(test_data)//window//batch_size

graph=tf.Graph()
with graph.as_default():
    wi_pvector=tf.placeholder(tf.float32,[None,window,embedding_size],name='wi_pvector')
    pvector=(wi_pvector+1e-10)/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(wi_pvector+1e-10),2)),2)#pvector normalization
    
    def generate_loss(y):#cross_entropy
        y=tf.reshape(y,[batch_size,window,embedding_size])
        corss_loss=0.
        for k in range(batch_size):
            loss=0.
            for i in range(window):
                object_word=tf.squeeze(tf.slice(y,[k,i,0],[1,1,embedding_size]),[0])
                if i==0:
                    context=tf.squeeze(tf.slice(y,[k,1,0],[1,window-1,embedding_size]))#[window-1,embedding_size]
                elif i==(window-1):
                    context=tf.squeeze(tf.slice(y,[k,0,0],[1,window-1,embedding_size]))
                else:
                    context=tf.concat([tf.squeeze(tf.slice(y,[k,0,0],[1,i,embedding_size]),[0]),tf.squeeze(tf.slice(y,[k,i+1,0],[1,window-i-1,embedding_size]),[0])],axis=0)
                weight=tf.matmul(object_word,context,transpose_b=True)#[1,window-1]
                pred=tf.matmul(weight,context)#[1,embedding_size]
                matrx=tf.concat([pred,context],axis=0)
                p=tf.nn.softmax(tf.matmul(object_word,matrx,transpose_b=True))[0][0]
                loss+=-tf.log(p+1e-10)
            corss_loss+=loss/window
        corss_loss/=batch_size
        return corss_loss

    cross_entropy=generate_loss(pvector)

def generate_input(step,corpus,dictionary):
#传入placeholder的数据必须为python scalars,str,list,numpy ndarray
    for j in range(batch_size):
        for k in range(window):
            p=np.array(w1[dictionary[corpus[step*batch_size*window+j*window+k]]])
            p.shape=(1,embedding_size)
            if k==0:
                new=p
            else:
                new=np.concatenate((new,p),axis=0)#[window,embedding_size]
        pvector1=new[np.newaxis]#[1,window,embdeeing_size]
        if j==0:
            pvector=pvector1
        else:
            pvector=np.concatenate((pvector,pvector1),axis=0)#[batch_size,window,embedding_size]
    return pvector 
    
with tf.Session(graph=graph,config=config) as session:
    print('初始化完毕')
    
    average_loss=0.
    for step in range(num_steps_train):
        w_p1=generate_input(step,train_data,dictionary)
        feed={wi_pvector:w_p1}
        [loss_val]=session.run([cross_entropy],feed_dict=feed)
        average_loss+=loss_val
    average_loss/=num_steps_train
    ppl=np.exp(average_loss)
    print('Train_loss: %.5f'%average_loss,' ppl:%.2f'%ppl)

    loss=0
    for step in range(num_steps_valid):
        w_p1=generate_input(step,valid_data,dictionary)
        feed={wi_pvector:w_p1}
        [loss_val]=session.run([cross_entropy],feed_dict=feed) 
        loss+=loss_val   
    loss/=num_steps_valid
    ppl=np.exp(loss)
    print('Vlid_loss: %.5f'%loss,' ppl:%.2f'%ppl)
        
    loss=0
    for step in range(num_steps_test):
        w_p1=generate_input(step,test_data,dictionary)
        feed={wi_pvector:w_p1}
        [loss_val]=session.run([cross_entropy],feed_dict=feed) 
        loss+=loss_val   
    loss/=num_steps_test
    ppl=np.exp(loss)
    print('Test_loss: %.5f'%loss,' ppl:%.2f'%ppl)