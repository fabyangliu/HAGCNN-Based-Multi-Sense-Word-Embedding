from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import pickle as pk
import pandas as pd
import numpy as np

config=tf.ConfigProto()
config.gpu_options.allow_growth=True#增长式

home_dir='/home/liu/word_sense_jungle'
#随机初始化哪一种label到对应filename找，vector到vectorname找
filename='random_initial_offset'
#text8_dataset wiki9_dataset
datafile='text8_dataset'
#text8 wiki9
dataname='text8'
#w2v_vector glove
vectorfile='w2v_vector'
vectorname='cboww20d200_text8.txt'

#预处理部分
#构造词典，将corpus转化为索引序列
def load_data():
    #读取pickle文件
    f1=open(home_dir+'/%s/corpus_en'%datafile,'rb')
    corpus=pk.load(f1)
    f1.close()
    f2=open(home_dir+'/%s/dict_en'%datafile,'rb')
    dictionary=pk.load(f2)
    f2.close()
    print('pickle读取完毕')
    return corpus,dictionary
corpus,dictionary=load_data()
#初始化词向量位置p矩阵  zh-wiki-decode1   vector-select_test  vector-100
w1 = KeyedVectors.load_word2vec_format(home_dir+'/%s/%s'%(vectorfile,vectorname), encoding='utf-8',binary=False)
print('初始化位置词向量矩阵p完毕')

#全局变量设置
vocab_size=len(dictionary)
#每个词多少种语义
n=10 
embedding_size=200

#构造label.csv
def generate_label(dictionary):
    label_p_po=[]
    for i in range(vocab_size):
        for j in range(n):
            label_p_po.append(dictionary[i]+'%d'%j)
    label_p_po=pd.DataFrame(label_p_po)
    label_p_po.to_csv(home_dir+'/%s/label_po_%d.csv'%(filename,n), header=False, index=False, encoding='utf-8')
generate_label(dictionary)

graph=tf.Graph()
with graph.as_default():
    p_vector=tf.placeholder(tf.float32, [vocab_size,embedding_size], name='positon_vector')
    #pvector[vocab_size,embedding_size] 归一化
    p_vector=(p_vector+1e-10)/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(p_vector+1e-10),1)),1)
        
    with tf.device('/cpu:0'):
        offset_embedding=tf.Variable(tf.random_normal(
                [vocab_size*n,embedding_size], 
                mean=0.0, 
                stddev=0.1,#offset 初始化标准差要来回试0.5  
                dtype=tf.float32))
    
    def generate_povector(p_vector,o_vector):
        for i in range(vocab_size):
            a=tf.slice(p_vector,[i,0],[1,embedding_size])
            b=tf.slice(o_vector,[i*n,0],[n,embedding_size])
            c=a+b#[n,embedding_size]
            if i==0:
                povector=c
            else:
                povector=tf.concat([povector,c],0)
        povector=(povector+1e-10)/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(povector+1e-10),1)),1)#归一化
        return povector
    
    input_data=generate_povector(p_vector,offset_embedding)
    init=tf.global_variables_initializer()
    
def generate_pvector(dictionary,w1):
    for x in range(vocab_size):
        p=np.array(w1[dictionary[x]])
        p.shape=(1,embedding_size)
        if x==0:
            pvector=p
        else:
            pvector=np.concatenate((pvector,p),axis=0)#[v,embedding_size]
    return pvector
    
with tf.Session(graph=graph,config=config) as session:
    init.run()
    labelp=pd.read_csv(home_dir+'/%s/label_po_%d.csv'%(filename,n),header=None,encoding='utf-8')
    labelp=labelp.T.values.tolist()[0]
    print('初始化完毕')
    
    pvector=generate_pvector(dictionary,w1)
    print('pvector已生成')
    
    [po_vector]=session.run([input_data],feed_dict={p_vector:pvector})
    print('povector已生成')
    
    for i in range(len(labelp)):
        v=' '.join('%.6f'%x for x in po_vector[i])
        l='%s'%(labelp[i])+' '
        #print(l)
        lne=l+v+'\n'
        if i==0:
            vector=lne
        else:
            vector=vector+lne
            
    f1=open(home_dir+'/random_initial_offset/po_%d.txt'%n,'w',encoding='utf-8')
    f1.write(vector)
    f1.close()
    print('povector已保存')