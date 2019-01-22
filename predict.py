#coding=utf-8

import numpy as np
import warnings
import pandas as pd
import math
from pandas import read_csv,DataFrame
from keras.models import Sequential
from keras.layers import Dense,LSTM,MaxPooling1D,Dropout,AveragePooling1D,Activation
from keras.layers.convolutional import Conv1D
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import time

def create_dataset(dataset,look_back):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)

def LSTM_model(dataset,epoch,look_back=10,LR=0.001):
    dataset=dataset.values
    dataset=dataset.astype('float32')
    #标准化。。。 先拟合数据，然后转化它将其转化为标准形式
    #scaler=MinMaxScaler(feature_range=(0,1),copy=True)
    scaler=StandardScaler(copy=True,with_mean=True,with_std=True)
    dataset=scaler.fit_transform(dataset)
    #划分训练集，测试集
    train_size=int(len(dataset)*0.9)
    #test_size=len(dataset)-train_size
    train=dataset[:train_size,:]
    test=dataset[train_size+1:,:]
    #test=dataset[train_size:train_size+120]
    #train,test=dataset[:9000,:],dataset[9000:9120,:]
    #reshape into X=t,Y=t+1,X是前t个时间点数据，Y是第t+1个点数据

    trainX,trainY=create_dataset(train,look_back)
    testX,testY=create_dataset(test,look_back)
    #print(len(testY))
    #将输入转换成[samples,times_steps,features]的形式
    trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
    testX=np.reshape(testX,(testX.shape[0],testX.shape[1],1,))
def build_model(layers):
    model = Sequential()
    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model


    #计算训练时间
    global_start_time = time.time()
    history=model.fit(trainX,trainY,epochs=epoch,batch_size=64,validation_split=0.05,verbose=1)
    print('Training duration (s) : ', time.time() - global_start_time)#打印训练时间

    #trainPredict = model.predict(trainX)
    Test_start = time.time()
    testPredict = model.predict(testX)
    print('Testing duration (s) : ', time.time() - Test_start)  # 打印测试时间
    #print(len(testPredict))
    #trainPredict = scaler.inverse_transform(trainPredict)
    #trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    testY=np.reshape(testY,(-1,1))
    #print(testY)

    # test_mse = model.test_on_batch(testPredict, testY)
    test_rmse = math.sqrt( mean_squared_error(testY[:,0], testPredict[:, 0]))
    test_mae=mean_absolute_error(testY[:,0], testPredict[:, 0])
    #print(testPredict.shape[0])
    #print(testY.shape[0])
    mape=MAPE(testPredict,testY)
    return test_rmse,testPredict,testY,history,mape,test_mae
def MAPE(predict,true):
    L1=int(true.shape[0])
    L2=int(predict.shape[0])
    #print(L1,   L2)
    if L1==L2:
        #SUM1=sum(abs(true-predict)/abs(true))
        SUM=0.0
        for i in range(L1-1):
            SUM=(abs(true[i,0]-predict[i+1,0])/true[i,0])+SUM
        per_SUM=SUM*100.0
        mape=per_SUM/L1
        return mape
    else:
        print("error")

def plt_result(y_test,predicted_values):
    fig = plt.figure()
    plt.plot(y_test)
    plt.plot(predicted_values)
    plt.xlabel('Time/5min')
    plt.ylabel('Electricity load (kWh)')
    plt.legend(['True', 'Predict'], loc='upper left')
    #plt.savefig('CNN_LSTM\house1\min5\house1_min5.jpeg', bbox_inches='tight')  # fig.savefig
    plt.show()



def save_result(y_test,predicted_values,history):
    tra_loss = history.history['loss']
    val_loss = history.history['val_loss']
    np.savetxt('CNN_LSTM\house1\min5\house1_min5_test.csv',y_test)
    np.savetxt('CNN_LSTM\house1\min5\house1_min5_predicted.csv',predicted_values)
    np.savetxt('CNN_LSTM\house1\min5\house1_min5_tra_loss.csv', tra_loss)
    np.savetxt('CNN_LSTM\house1\min5\house1_min5_val_loss.csv', val_loss)


np.random.seed(7)
#加载数据
dataframe=read_csv('data.csv',header=None)#hour_data\house5_hour.csv

warnings.filterwarnings("ignore")#运行时忽略warning
model=build_model([1, 50, 100, 1])
model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

#画图
plt_result(y_test,predicted_values)
#保存结果
#save_result(y_test,predicted_values,history)
#打印评价指标
print('RMSE:',test_rmse)
print('MAPE:',mape)
print('MAE:',MAE)

