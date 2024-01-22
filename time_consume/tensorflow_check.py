# test time consum
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from time import time

#  plot_demo(l_sizes,t_all_o,t_all_n,"unit_size",u_sizes)
def plot_demo(xs,t_dic_o,t_dic_n,exp,x_label,cs=["blue","red"]):
    print(x_label)
    for k in xs:
        t_o=t_dic_o[k]
        plt.plot(range(len(x_label)),t_o,c=cs[0],label="gdcn layers={}".format(k))
        t_n=t_dic_n[k]
        plt.plot(range(len(x_label)),t_n,c=cs[1],label="gdcnv2 layers={}".format(k))
    plt.xlabel("{}".format(exp))
    plt.ylabel("consuming time")
    plt.xticks(range(len(x_label)),x_label)
    plt.legend()
    plt.savefig("{}.png".format(exp))

def ori_gdcn(a,l,u):
    b=time()
    a0=a
    for li in range(l):
        a=tf.keras.layers.Dense(
                            units=u,
                            use_bias=True,
                            trainable=False)(a)
        g=tf.keras.layers.Dense(
                            units=u,
                            use_bias=False,
                            trainable=False)(a)
        a=a0*a*tf.math.sigmoid(g)+a
    #sess=tf.Session()
    #sess.run(a)
    e=time()
    print("{0} and {1} old consum {2}".format(l,u,e-b))
    return l,u,e-b

def new_ldcn(a,l,u):
    b=time()
    
    if l%2==0:
        flag=True
    else:
        flag=False
    t_l=int(math.log(l,2))
    if t_l<math.ceil(math.log(l,2)):
        t_l+=1
    
    for li in range(t_l):
        a_=tf.keras.layers.Dense(
                            units=u,
                            use_bias=True,
                            trainable=False)(a)
        
        a=a*a_+a
    g=tf.keras.layers.Dense(
                            units=u,
                            use_bias=False,
                            trainable=False)(a)
    
    if flag:
        a_=tf.keras.layers.Dense(
                            units=u,
                            use_bias=True,
                            trainable=False)(a)
        
        a=a*a_*tf.math.sigmoid(a) + a
    else:
        a=a*tf.math.sigmoid(a)+a
    
    #sess=tf.Session()
    #sess.run(a)
    e=time()
    print("{0} and {1} new consum {2}".format(l,u,e-b))
    return l,u,e-b

if __name__=="__main__":
    b=1024
    u_sizes=[32,64,128,256,512,1024]
    l_sizes=[3,5,7,9,11,15]
    t_all_o=dict(zip(l_sizes,[[] for _ in range(len(l_sizes))]))
    t_all_n=dict(zip(l_sizes,[[] for _ in range(len(l_sizes))]))
#     print(t_all_o)
    for u in u_sizes:
        for l in l_sizes: 
            a=tf.ones((b,u),dtype=tf.float32)
            l_o,u_o,t_o=ori_gdcn(a,l,u)
            l_n,u_n,t_n=new_ldcn(a,l,u)
            t_all_o[l].append(t_o)
            t_all_n[l].append(t_n)
    plot_demo(l_sizes,t_all_o,t_all_n,"unit_size",u_sizes)

    
    
