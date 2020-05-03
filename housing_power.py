# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold
import decimal
decimal.getcontext().prec = 8

data=pd.read_csv(r"C:\Users\ISHIKA\4th\household_power_consumption\household_power_consumption.txt",delimiter=";",parse_dates={"datetime":["Date","Time"]},index_col='datetime')

#print(data.head())
#data.drop(["Date","Time"],axis=1,inplace=True)




cols=data.columns
data[cols]=data[cols].replace(["?"],[None])
data[cols]=data[cols].astype(float)
data=data.fillna(data.mean(axis=1))
data.dropna(inplace=True)
data.plot(subplots=True,legend=True)
plt.show()





#data.replace(["?"],[data.mean()],inplace=True,axis=1)
#print(data.head())
x=data.drop(['Global_active_power'],axis=1)



y=data[['Global_active_power']]






x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7,random_state=200)
x_train=x_train.to_numpy()
x_test=x_test.to_numpy()
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()
from sklearn.preprocessing import StandardScaler
#x_train=x_train.to_numpy()
#x=x.astype(float)
scaler=StandardScaler()
scaler.fit(x_train)
scaler.fit(y_train)
scaler.fit(x_test)
scaler.fit(y_test)
x_train=scaler.transform(x_train)
y_train=scaler.transform(y_train)
x_test=scaler.transform(x_test)
y_test=scaler.transform(y_test)
#t=Normalizer()
#x=t.transform(x)    
np.nan_to_num(x_train)
np.nan_to_num(x_test)




from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=4,include_bias=True)
x_poly=pr.fit_transform(x_train)
#y_Poly=pr.transform(y_test)
pr.fit(x_poly,y_train)
print("checklist2")
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
print("checklist3")
lr.fit(x_poly,y_train)
print("checklist4")

plt.scatter(x[["Sub_metering_1"]],y,color="Red")
plt.scatter(x[["Global_reactive_power"]],y,color="pink")
plt.scatter(x[["Voltage"]],y,color="blue")
plt.scatter(x[["Global_intensity"]],y,color="green")
plt.scatter(x[["Sub_metering_2"]],y,color="black")

plt.scatter(x[["Sub_metering_3"]],y,color="yellow")
plt.show()
plt.rcParams['agg.path.chunksize'] = 10000
y_pred=lr.predict(pr.fit_transform(x_test))
plt.plot(x_test,lr.predict(pr.fit_transform(x_test)),color="black")
plt.show()
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
accu=accuracy_score(y_test,y_pred)
plt.plot(x_test,y_test-y_pred,color="Red")
s


