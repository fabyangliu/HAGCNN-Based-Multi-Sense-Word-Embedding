# coding: utf-8
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import numpy as np
import collections
import pickle as pk
import pandas as pd
#import os

#os.environ['CUDA_VISIBLE_DEVICES']='0'#指定使用第一块GPU
config=tf.ConfigProto()
config.gpu_options.allow_growth=True#增长式
#config.gpu_options.per_process_gpu_memory_fraction=0.9#bai fen bi mo shi

home_dir='/home/liu/word_sense_jungle'
save_dir=home_dir+'/logs/save'
log_dir=home_dir+'/logs/summaries'
filename='300'
#text8_dataset wiki9_dataset
datafile='text8_dataset'
#text8 wiki9
dataname='text8'
#w2v_vector glove
vectorfile='w2v_vector'
#注意是否为二进制，后面代码要相应修改
vectorname='cboww20d300_text8.txt'

#预处理部分
#构造词典，将corpus转化为索引序列
def build_dataset():
    fin=open(home_dir+'/%s/%s'%(datafile,dataname),'r',encoding='utf-8')
    line=fin.read()
    line_f=line.split(' ')
    print('读取语料完毕')
    #释放空间
    del line
    #while ''in line_f:
        #line_f.remove('')
    fin.close()
    
    count=[['OOV',0]]
    #前面是word后面是频率
    count.extend(collections.Counter(line_f).most_common())
    dictionary=dict()
    for word,_ in count:
        if _ >=5:
            dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    for word in line_f:
        if word in dictionary:
            index=dictionary[word]
            data.append(index)
        else:
            unk_count+=1
    print('Unknown words: %d \nDictionary length: %d \nCorpus length: %d \n' % (unk_count,len(dictionary),len(data)))   
    dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    
    #将corpus，dictionary保存为pickle
    f1=open(home_dir+'/%s/corpus_en'%datafile,'wb')
    pk.dump(data,f1,2)
    f1.close()
    f2=open(home_dir+'/%s/dict_en'%datafile,'wb')
    pk.dump(dictionary,f2,2)
    f2.close()
    print('pickle保存完毕')
    return data,dictionary

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

#corpus,dictionary=build_dataset()
corpus,dictionary=load_data()
    
#初始化词向量位置p矩阵
w1 = KeyedVectors.load_word2vec_format(home_dir+'/%s/%s'%(vectorfile,vectorname), encoding='utf-8',binary=False)
print('初始化位置词向量矩阵p完毕')

#全局变量设置
vocab_size=len(dictionary)
#每个词多少种语义
n=4 
window=20
embedding_size=300
#可以尝试一下看看影响
drop_keep=0.75
batch_size=48
num_steps=len(corpus)//window//batch_size

#构造label.csv
def generate_label(dictionary):
    label=pd.DataFrame(list(dictionary.values()),index=dictionary.keys())
    label.to_csv(home_dir+'/%s/label_p_en.csv'%filename, header=False, index=False, encoding='utf-8')
    label_p_po=[]
    for i in range(vocab_size):
        for j in range(n):
            label_p_po.append(dictionary[i]+'%d'%j)
    label_p_po=pd.DataFrame(label_p_po)
    label_p_po.to_csv(home_dir+'/%s/label_po_%d.csv'%(filename,n), header=False, index=False, encoding='utf-8')
#generate_label(dictionary)


def weight_variable(shape,x,p,nl):
    if x==1:
        #第一层
        initial=tf.random_normal(
            shape,
            mean=0.0,
            stddev=tf.sqrt(4*p/nl),
            dtype=tf.float32,
            seed=None,
            name='weight%d'%x)
    else:
        #第二层
        initial=tf.random_normal(
            shape,
            mean=0.0,
            stddev=tf.sqrt(4*p/nl),
            dtype=tf.float32,
            seed=None,
            name='weight%d'%x)
    return tf.Variable(initial)

def bias_variable(shape,x):
    if x==1:
        #未使用relu
        initial=tf.constant(0.0,shape=shape)
    else:
        #使用relu
        initial=tf.constant(0.05,shape=shape)
    return tf.Variable(initial)
    
def attention_gated_cnn(x,w,bias,num):
    '''
    cnn+glu,输入尺寸为n×d，卷积核为（n+1）×d×2d，偏置为n×2d（n应为偶数）
    卷积词时n=8，卷积句子时n=10,20这样的偶数
    首先对输入数据首尾各补n/2的0向量（n/2×d），变成一个2n×d的向量
    '''
    #x [batch_size,num,embedding_size,1]单通道
    #首尾各补num/2的0向量
    pad=tf.zeros([batch_size,num//2,embedding_size,1],tf.float32)
    x=tf.concat([pad,x],1)
    x=tf.concat([x,pad],1)
    conv=tf.nn.conv2d(
        x,
        w,
        strides=[1,1,1,1],
        padding='VALID',
        name='conv')
    #conv是[batch_size,num,1,2d],降维到[batch_size,num,2d]转化成[2d,num,batch_size]
    conv=tf.transpose(tf.squeeze(tf.nn.bias_add(conv,bias)))
    #拆分成A、B两个部分
    a,b=tf.split(conv,2,0, name='split')
    #B部分sigmoid
    b_sigmoid=tf.nn.sigmoid(b)
    #A、B部分逐点乘,A*B直接逐点乘
    c=a*b_sigmoid
    #return矩阵C[d,num,batch_size]
    return c
        
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

graph=tf.Graph()
with graph.as_default():
    wi=tf.placeholder(tf.int32,[None,n*window],name='wi')
    with tf.name_scope('wi-pvector'):
        wi_pvector=tf.placeholder(tf.float32,[None,n*window,embedding_size],name='wi_pvector')
        #pvector=wi_pvector
        pvector=(wi_pvector+1e-10)/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(wi_pvector+1e-10),2)),2)#pvector normalization
        variable_summaries(pvector, 'position-vector')

    keep_prob=tf.placeholder(tf.float32,name='keep_prob')
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.5

    #词级别选择语义参数，以后可能要深层
    with tf.name_scope('w_conv1'):
        w_conv1=weight_variable([n+1,embedding_size,1,2*embedding_size],1,drop_keep,(n+1)*embedding_size)
        variable_summaries(w_conv1, 'weight-conv1')
    with tf.name_scope('b_conv1'):
        b_conv1=bias_variable([2*embedding_size],0)
        variable_summaries(b_conv1, 'bias-conv1')

    #句级别选择语义参数，以后可能要深层
    with tf.name_scope('w_conv5'):
        w_conv5=weight_variable([window+1,embedding_size,1,2*embedding_size],1,drop_keep,(window+1)*embedding_size)
        variable_summaries(w_conv5, 'weight-conv5')
    with tf.name_scope('b_conv5'):
        b_conv5=bias_variable([2*embedding_size],0)
        variable_summaries(b_conv5, 'bias-conv5')
        
    with tf.device('/cpu:0'):
        offset_embedding=tf.Variable(tf.random_normal(
                [vocab_size*n,embedding_size], 
                mean=0.0, 
                stddev=0.1,#offset 初始化标准差要来回试0.5  
                dtype=tf.float32),name='offset_embedding')
        wi_ovector_fuck=tf.nn.embedding_lookup(offset_embedding,wi)#[batch_size*window*n,embedding_size]
    #input是一个np矩阵，都是word索引,返回[batch_size,sequence_length,embedding_size]
    #window窗口内的词偏移向量矩阵wi[]
    with tf.name_scope('wi_ovector'):
        wi_ovector=tf.reshape(wi_ovector_fuck,[batch_size,window*n,embedding_size])
        variable_summaries(wi_ovector, 'offset-vector')
    del wi_ovector_fuck

    #生成p+o向量矩阵[batch_size,n*window,50,1]
    with tf.name_scope('povector'):
        povector_=pvector+wi_ovector#[batch_size,n*window,embedding_size]
        #povector=tf.expand_dims(povector_,-1)
        povector=tf.expand_dims((povector_+1e-10)/tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(povector_+1e-10),2)),2),-1)#pvector normalization
        variable_summaries(povector_, 'position-offset-vector')
    
    
    #卷积输入层dropout
    povector_drop=tf.nn.dropout(povector,keep_prob)
    
    def cnn_batch_normal(x):
        #x [batch_size,window,embedding_size]
        batch_mean=tf.expand_dims(tf.reduce_mean(x,2),-1)#[batch_size,window,1]
        batch_var=tf.sqrt(tf.expand_dims(tf.reduce_sum(tf.square(x-batch_mean),2),-1)/(embedding_size)+0.001)
        normalized_x=(x-batch_mean)/(batch_var+1e-10)
        return normalized_x#[batch_size,window,embedding_size]

    #第一层,先对window个povector分别glu，然后接连起来
    def generate_matrix1_x(povector,w_conv,b_conv):
        #povector[batch_size,n*window,embedding_size,1]
        for i in range(window):
            x=tf.slice(povector,[0,i*n,0,0],[batch_size,n,embedding_size,1])
            c_i=tf.transpose(cnn_batch_normal(tf.transpose(attention_gated_cnn(x,w_conv,b_conv,n))))#[d,n,batch_size]
            if i==0:
                c_x=c_i
            else:
                c_x=tf.concat([c_x,c_i],1)
                #c_x[d,n*window,batch_size]    
        return c_x
    with tf.name_scope('matrix1-conv'):
        c_x=generate_matrix1_x(povector_drop,w_conv1,b_conv1)
        variable_summaries(c_x, 'matrix1-conv-out')
    #c_x[d,n*window,batch_size]
    del povector_drop
    
    #decoder层dropout
    c_x_drop=tf.nn.dropout(c_x,keep_prob) 
    #矩阵乘法得到[batch_size,n*window,n*window,1]，maxpooling，加权求和
    
    def get_diag(m):
        #m [batch_size,n*window,n*window]
        m=tf.reshape(m,[batch_size,n*window,n*window])
        for j in range(batch_size):
            x=tf.expand_dims(tf.diag(tf.diag_part(m[j])),0)#[1,n*window,n*window]
            if j==0:
                y=x
            else:
                y=tf.concat([y,x],0)
        return y
    
    def max_pool(x,h):
        #x是[batch_size,n*window,n*window,1]，返回[batch_size,1,n*window,1] 
        a=tf.nn.max_pool(x,ksize=[1,h,1,1],strides=[1,1,1,1],padding='VALID',name='max-pooling-weight')
        return a
    def batch_normal(a,size):
        #权重归一化
        #最后一维度每一行求平均,[batch_size,size,size]对整个batch运算
        mean_batch_=tf.reduce_sum(a,[1,2])/(size*(size-1))#[batch_size]
        for i in range(batch_size):
            x=tf.expand_dims(mean_batch_[i],0)#[1]
            for j in range(size):
                if j==0:
                    mean_batch=x
                else:
                    mean_batch=tf.concat([mean_batch,x],0)#[size]
            m_mean=tf.expand_dims(mean_batch,0)#[1,size]
            window_mean=tf.expand_dims(tf.diag(mean_batch),0)#[1,size,size]
            if i==0:
                batch_mean=window_mean
                match_mean=m_mean
            else:
                batch_mean=tf.concat([batch_mean,window_mean],0)#[batch_size,size,size]对角线
                match_mean=tf.concat([match_mean,m_mean],0)#[batch_size,size]整个空间
        match_mean=tf.expand_dims(match_mean,1)#[batch_size,1,size]
        #batch_mean=tf.squeeze(batch_mean,[2])
        a+=batch_mean
        #求variance
        var_batch=tf.sqrt(tf.reduce_sum(tf.square(a-match_mean),[1,2])/(size*(size-1))+0.001)#[batch_size]
        for j in range(batch_size):
            x=tf.expand_dims(var_batch[j],0)#[1]
            for i in range(size):
                if i==0:
                    var_=x
                else:
                    var_=tf.concat([var_,x],0)#[n*window]
            var_=tf.expand_dims(var_,0)#[1,n*window]
            if j==0:
                var_m=var_
            else:
                var_m=tf.concat([var_m,var_],0)#[batch,n*window]
        var_batch=tf.expand_dims(var_m,1)#[batch,1,n*window]
        return match_mean,var_batch,a
    
    def attention_word_matrix(c_x_drop,c_y_in):
        #c_x[d,n*window,batch_size]
        for i in range(batch_size):
            c_x=tf.squeeze(tf.slice(c_x_drop,[0,0,i],[embedding_size,n*window,1]))
            c_y=tf.squeeze(tf.slice(c_y_in,[0,0,i],[embedding_size,n*window,1]))
            matrx1_=tf.expand_dims(tf.matmul(c_x,c_y,transpose_a=True,transpose_b=False,name='attention-word-matrix-%d' % i),0)
            #[1,n*window,n*window]
            if i==0:
                a=matrx1_
            else:
                a=tf.concat([a,matrx1_],0)#[batch_size,n*window,n*window]
        dia=get_diag(a)#[batch_size,n*window,n*window]
        a-=dia#对角线为0
        match_mean,var_batch,a=batch_normal(a,n*window)
        #可以在这个地方，（a-match_mean) / var_batch
        #normalized_weight_x[batch_size,n*window,n*window]
        normalized_weight_x=(a-match_mean)/(var_batch+1e-10)
        #batchnormal
        normalized_weight_x=tf.expand_dims(normalized_weight_x,-1)#[batch_size,n*window,n*window,1]
        b=max_pool(normalized_weight_x,n*window)#[batch_size,1,n*window,1]
        return b
    with tf.name_scope('word-attention-matrix'):
        weight_vector1=attention_word_matrix(c_x_drop,c_x)
        variable_summaries(weight_vector1, 'word-attention-matrix')
    del c_x_drop
    del c_x
    #[batch_size,1,n*window,1]

    weight_vector=tf.reshape(tf.squeeze(weight_vector1),[batch_size,window,n])#[batch_size,window,n]
    del weight_vector1
    
    def generate_piovector(c_y_,weight_vector):
        #c_y_[batch_size,n*window,embedding_size]
        #weight_vector[batch_size,window,n]
        c_y=tf.transpose(
                tf.reshape(
                    c_y_,
                    [batch_size,n,embedding_size*window]),
                [1,2,0])
        #c_y[n,embedding_size*window,batch_size]
        #normalize weight
        abs_sum_weight=tf.expand_dims(tf.reduce_sum(tf.abs(weight_vector),reduction_indices=2),-1)+1e-10#[batch_size,window,1]
        weight_vector=weight_vector/abs_sum_weight
        for j in range(batch_size):
            for i in range(window):
                #c_y_slce[n,d]
                c_y_slce=tf.squeeze(tf.slice(c_y,[0,embedding_size*i,j],[n,embedding_size,1]))
                #normalized_weight[1,n]
                normalized_weight=tf.squeeze(tf.slice(weight_vector,[j,i,0],[1,1,n]),[0])
                #po[1,d]
                po=tf.matmul(normalized_weight,c_y_slce,transpose_a=False,transpose_b=False)
                #po_1[1,1,d]
                po_1=tf.expand_dims(po,0)
                if i==0:
                    #po_vector[1,1,d]
                    po_vector=po_1
                else:
                    #po_vector[1,window,d]
                    po_vector=tf.concat([po_vector,po_1],1)
            if j==0:
                po_c_vector=po_vector
            else:
                #po_c_vector[batch_size,window,d]
                po_c_vector=tf.concat([po_c_vector,po_vector],0)
        po_c_vector=tf.expand_dims(po_c_vector,-1)
        #po_c_vector [batch_size,window,d,1]
        return po_c_vector
    #povector_[batch_size,n*window,embedding_size]
    #weight_vector[batch_size,window,n]
    with tf.name_scope('pio-vector'):
        pio_vector_cnn=generate_piovector(povector_,weight_vector)
        variable_summaries(pio_vector_cnn, 'pio-word-vector')
    del weight_vector
    
    #第二层,卷积之后可以尝试多层
    #卷积输入dropout
    #po_c_vector [batch_size,window,d,1]
    pio_vector_cnn_drop=tf.nn.dropout(pio_vector_cnn,keep_prob)
    #with tf.name_scope('matrix2-conv'):
        #c_pio_=tf.expand_dims(cnn_batch_normal(tf.transpose(attention_gated_cnn(pio_vector_cnn_drop,w_conv2,b_conv2,window))),-1)
        #variable_summaries(c_pio_, 'matrix2-conv-out')
    #c_pio_ [batch_size,window,d,1]
    #del pio_vector_cnn_drop
    #decoder层drop
    #c_pio_drop_=tf.nn.dropout(c_pio_,keep_prob)
    #del c_pio_
    #2nd 2nd cnn
    #with tf.name_scope('matrix3-conv'):
        #c_pio3=tf.expand_dims(cnn_batch_normal(tf.transpose(attention_gated_cnn(c_pio_drop_,w_conv3,b_conv3,window))),-1)
        #variable_summaries(c_pio3, 'matrix3-conv-out')
    #c_pio [batch_size,window,d,1]
    #del c_pio_drop_
    #decoder层drop
    #c_pio_drop3=tf.nn.dropout(c_pio3,keep_prob)  
    #del c_pio3  
    #2nd 3nd cnn
    #with tf.name_scope('matrix4-conv'):
        #c_pio4=tf.expand_dims(cnn_batch_normal(tf.transpose(attention_gated_cnn(c_pio_drop3,w_conv4,b_conv4,window))),-1)
        #variable_summaries(c_pio4, 'matrix4-conv-out')
    #c_pio [batch_size,window,d,1]
    #del c_pio_drop3
    #decoder层drop
    #c_pio_drop4=tf.nn.dropout(c_pio4,keep_prob)  
    #del c_pio4  
    #2nd 4nd cnn    
    with tf.name_scope('matrix5-conv'):
        c_pio=tf.transpose(cnn_batch_normal(tf.transpose(attention_gated_cnn(pio_vector_cnn_drop,w_conv5,b_conv5,window))))
        variable_summaries(c_pio, 'matrix5-conv-out')
    #c_pio [d,window,batch_size]
    del pio_vector_cnn_drop
    #decoder层drop
    c_pio_drop=tf.nn.dropout(c_pio,keep_prob)  
    
    def attention_sentence_matrix(c_object,c_context):
        #c_context[d,window,batch_size]
        c_object=tf.transpose(c_object)
        #c_object[batch_size,window,d]
        for i in range(batch_size):
            c_object_slce=tf.squeeze(tf.slice(c_object,[i,0,0],[1,window,embedding_size]))#[window,d]
            c_context_slce=tf.squeeze(tf.slice(c_context,[0,0,i],[embedding_size,window,1]))#[d,window]
            matrix=tf.matmul(c_object_slce,c_context_slce,transpose_a=False,transpose_b=False,name='attention-sentence-matrix')    
            m=tf.expand_dims(matrix,0)
            #[1,window,window]
            if i==0:
                matrx1=m
            else:
                matrx1=tf.concat([matrx1,m],0)#[batch_size,window,window]
        return matrx1
    #matrix2 [batch_size,window,window]
    with tf.name_scope('sentence-attention-matrix'):
        matrix2=attention_sentence_matrix(c_pio_drop,c_pio)
        variable_summaries(matrix2, 'sentence-attention-matrix')
    del c_pio
    del c_pio_drop

    
    #pred单词向量，及预测层
    def normalize_pred(matrx,c_po):
    #matrx [batch_size,window,window]
    #c_po [batch_size,window,d,1]
        c_po=tf.transpose(tf.squeeze(c_po),perm=[1,2,0])
        #c_po [window,d,batch_size]
        #dia=np.zeros((batch_size,window,window),dtype=np.int32)
        for k in range(batch_size):
            x=tf.expand_dims(tf.diag(tf.diag_part(matrx[k])),0)#[1,window,window]
            if k==0:
                y=x
            else:
                y=tf.concat([y,x],0)
        matrx-=y#[batch_size,window,window]
        
        match_mean,var_batch,matrx=batch_normal(matrx,window)
        normalized_weight_x=(matrx-match_mean)/(var_batch+1e-10)#归一化后，[batch,window,window]
        
        abs_sum_weight=tf.expand_dims(tf.reduce_sum(tf.abs(normalized_weight_x),reduction_indices=2),-1)+1e-10#[batch_size,window,1]
        normalzed_matrx2=normalized_weight_x/abs_sum_weight#【batch_size,window,window】
        
        #normalzed_matrx2 [batch_size,window,window]
        for j in range(batch_size):
            c_po_slce=tf.squeeze(tf.slice(c_po,[0,0,j],[window,embedding_size,1]))#[window,d]
            weight_slice=tf.squeeze(tf.slice(normalzed_matrx2,[j,0,0],[1,window,window]),[0])#[window,window]
            po=tf.matmul(weight_slice,c_po_slce,transpose_a=False,transpose_b=False)#[window,d]
            po_1=tf.expand_dims(po,0)#[1,window,d]
            if j==0:
                po_vector=po_1#[1,window,d]
            else:
                po_vector=tf.concat([po_vector,po_1],1)#[batch_size,window,d]
        return po_vector
    with tf.name_scope('normalize-weight-prediction-word-vector'):
        pred_vector=normalize_pred(matrix2,pio_vector_cnn)
        variable_summaries(pred_vector, 'normalize-weight-prediction-word-vector')
    
    def generate_prediction_matrix(pred,c):
    #pred [batch_size,window,d]
    #c [batch_size,window,d,1]
        pred=tf.reshape(pred,[batch_size,window,embedding_size])+1e-10
        c=tf.reshape(c,[batch_size,window,embedding_size])+1e-10
        for i in range(batch_size):
            a=tf.slice(pred,[i,0,0],[1,window,embedding_size])
            a=tf.reshape(a/tf.expand_dims(tf.reduce_sum(tf.square(a),2),2),[window,embedding_size])
            b=tf.slice(c,[i,0,0],[1,window,embedding_size])
            b=tf.reshape(b/tf.expand_dims(tf.reduce_sum(tf.square(b),2),2),[window,embedding_size])
            c_out_=tf.expand_dims(tf.matmul(a,b,transpose_a=False,transpose_b=True),0)#[1,window,window]
            if i==0:
                c_out=c_out_
            else:
                c_out=tf.concat([c_out,c_out_],0)
        #c_out+=1e-10
        #[batch_size,window,window]   
        for j in range(window):
            c_slice=tf.slice(c_out,[0,j,0],[batch_size,1,window])
            c_slice=tf.nn.softmax(c_slice)#[batch,1,window]
            if j==0:
                window_out=c_slice
            else:
                window_out=tf.concat([window_out,c_slice],1)#[batch_size,window,window]
        window_out+=1e-10
        return window_out#[batch_size,window,window]
    with tf.name_scope('prediction-propability-softmax'):
        y_pred=generate_prediction_matrix(pred_vector,pio_vector_cnn)
        variable_summaries(y_pred, 'prediction-propability')

    #pred_vector [batch_size,window,d]
    #po_vector_cnn [batch_size,window,d,1]
    del pio_vector_cnn
    del matrix2
    del pred_vector
    
    def generate_loss(y):#cross_entropy
    #y [batch_size,window,window] 
        y=tf.reshape(y,[batch_size,window,window])
        x=0.
        for j in range(batch_size):
            x-=tf.reduce_mean(tf.log(tf.diag_part(y[j])+1e-10))#因为有些预测值为0
        x=x/batch_size
        return x

    with tf.name_scope('loss'):
        cross_entropy=generate_loss(y_pred)
        tf.summary.scalar('loss', cross_entropy)
    del y_pred
 
    #optimizer
    #MomentumOptimizer(learning_rate=0.25, momentum=0.99,use_locking=False, #True不会对variable更新name="Nesterov-Momentum", use_nesterov=True)可以用下降太慢不明显
    #.GradientDescentOptimizer(learning_rate=)可以用
    #.AdamOptimizer(1e-4)adam机子直接死掉
    #lr decay
    lr = tf.train.exponential_decay(starter_learning_rate, global_step,2000, 0.85, staircase=True)
    optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.99,use_locking=False,name='nesterov',use_nesterov=True)#.minimize(cross_entropy)
    #optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5
    grads = optimizer.compute_gradients(cross_entropy)
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] =[tf.clip_by_norm(g, 0.035), v]  # clip gradients
    train_op = optimizer.apply_gradients(grads, global_step=global_step)#minimize(cross_entropy)
    
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    merged = tf.summary.merge_all()

def generate_input(step,corpus,dictionary):
#传入placeholder的数据必须为python scalars,str,list,numpy ndarray
    v=np.zeros((batch_size,n*window),dtype=np.int32)
    for j in range(batch_size):
        for k in range(window):
            p=np.array(w1[dictionary[corpus[step*batch_size*window+j*window+k]]])
            p.shape=(1,embedding_size)
            for m in range(n):
                v[j][k*n+m]=corpus[step*batch_size*window+j*window+k]+m
                if (m==0):
                    p_n=p
                else:
                    p_n=np.concatenate((p_n,p),axis=0)#[n,embedding_size]
            if k==0:
                new=p_n
            else:
                new=np.concatenate((new,p_n),axis=0)#[n*window,embedding_size]
        pvector1=new[np.newaxis]#[1,n*window,embdeeing_size]
        if j==0:
            pvector=pvector1
        else:
            pvector=np.concatenate((pvector,pvector1),axis=0)#[batch_size,n*window,embedding_size]
    v=np.array(v,dtype=np.int32)
    return pvector,v
#v是np.array [batch_size,n*window]
#pvector是np.array [batch_size,n*window,embedding_size]  
def generate_pvector(dictionary,w1):
    for x in range(vocab_size):
        p=np.array(w1[dictionary(x)])
        p.shape=(1,embedding_size)
        if x==0:
            pvector=p
        else:
            pvector=np.concatenate((pvector,p),axis=0)#[v,embedding_size]
    return pvector   
    
#,config=config
with tf.Session(graph=graph,config=config) as session:
    train_writer=tf.summary.FileWriter(log_dir+'/train',session.graph)
    init.run()
    print('初始化完毕')
        
    average_loss=0
    per_loss=0
    global_loss=0
    for step in range(num_steps):
        global_step=step
        w_p1,v1=generate_input(step,corpus,dictionary)
        #v1是np.array [batch_size,n*window]
        #w_p1是np.array [batch_size,n*window,embedding_size]

        feed={wi:v1,wi_pvector:w_p1,keep_prob:drop_keep}
        
        _,loss_val,summary=session.run([train_op,cross_entropy,merged],feed_dict=feed)
        
        train_writer.add_summary(summary,step)
        
        per_loss+=loss_val
        global_loss+=loss_val

        print('第%d步，loss：%.5f'%(step,loss_val))
        
        if ((step+1)%100==0)and(step>0):
            per_loss/=100
            print('第%d步'%step,'mini-batch-loss: ',per_loss)
            per_loss=0
            
        average_loss+=loss_val
        
        if ((step+1)%500==0)and(step>0):
            average_loss/=500
            print('每500步平均残差,目前在第',step,'步: %.5f'%average_loss)
            average_loss=0
        
        #每2000步后保存模型
        if ((step+1)%2000==0)and(step>0):
            global_loss/=(step+1)
            ppl=np.exp(global_loss)
            save_path = saver.save(session, save_dir+"/model_step%dloss%.5fppl%.5f.ckpt" % (step,global_loss,ppl))    
    save_path = saver.save(session, save_dir+"/model_step%d.ckpt"% step)
    print("训练完毕模型存储在该路径中: ", save_path)
#到命令行执行 tensorboard --logdir=/home/liu/word_sense_jungle/logs/summaries
#出来提示信息的网页，复制到浏览器即可
#恢复变量restore就行，saver.restore(sess, "/tmp/model.ckpt")
#如果只恢复某个变量到某个变量，则  saver = tf.train.Saver({"my_v2": v2}) 前者是模型变量中名称