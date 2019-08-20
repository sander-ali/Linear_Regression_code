#Boston Housing Price Dataset 
#Linear Regression Example

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target

ax, fig = plt.subplots(figsize = (16,10))
sns.heatmap(df.corr(), annot = True, cmap = 'RdBu')
plt.show()

f = plt.Figure(figsize= (8,6))
pd.plotting.scatter_matrix(df, figsize= (20,20), s = 75)
plt.plot()

def normal_equation(x,y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)


X_b = np.c_[np.ones((len(df),1)),df.values]
X_b.shape

from sklearn.model_selection import train_test_split
X = pd.DataFrame(np.c_[df['LSTAT'], df['RM'], df['CRIM'], df['ZN'], df['INDUS'], df['PTRATIO'],df['TAX'], df['B']] , columns = ['LSTAT','RM', 'CRIM','ZN','INDUS','PTRATIO','TAX','B'])
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)

theta = normal_equation(x_train,y_train)

predictions = np.dot(x_train,theta)
ax,fig = plt.subplots(figsize=(10,8))
plt.plot(predictions,'b.',marker = '*')
plt.plot(y_train,'r.')
plt.legend(['predictions','true'])
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(predictions,y_train)

test_pred = np.dot(x_test,theta)
ax,fig = plt.subplots(figsize=(10,8))
plt.plot(test_pred,'b.',marker = '*')
plt.plot(y_test,'r.')
plt.legend(['predictions','true'])
plt.show()
mean_squared_error(test_pred,y_test)

fig, axes = plt.subplots(2,1, figsize = (10,15))
axes[0].plot(x_test.values[:,1],y_test,'b.')
axes[0].plot(x_test.values[:,1],test_pred,'r.')
axes[0].plot(np.unique(x_test.values[:,1]), np.poly1d(np.polyfit(x_test.values[:,1], y_test.reshape(-1),1))(np.unique(x_test.values[:,1])),'g')
axes[0].plot(np.unique(x_test.values[:,1]), np.poly1d(np.polyfit(x_test.values[:,1], test_pred.reshape(-1),1))(np.unique(x_test.values[:,1])),'r')
axes[0].legend(['true','preds','true line','pred line'], loc = 'best')
axes[0].set_xlabel('ROOMS')
axes[0].set_ylabel('MEDV')
axes[1].plot(x_test.values[:,0],y_test,'b.',)
axes[1].plot(x_test.values[:,0],test_pred,'r.')
axes[1].plot(np.unique(x_test.values[:,0]), np.poly1d(np.polyfit(x_test.values[:,0], y_test.reshape(-1),1))(np.unique(x_test.values[:,0])),'g')
axes[1].plot(np.unique(x_test.values[:,0]), np.poly1d(np.polyfit(x_test.values[:,0], test_pred.reshape(-1),1))(np.unique(x_test.values[:,0])),'r')
axes[1].legend(['true','preds','true line','pred line'], loc = 'best')
axes[1].set_xlabel('ROOMS')
axes[1].set_ylabel('MEDV')
plt.show()

from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(x_train, y_train)
# model evaluation for training set
y_train_predict = lin_model.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

rmse_mine = (np.sqrt(mean_squared_error(y_test,test_pred)))
r2_mine = r2_score(y_test,test_pred)

print("The model performance for my code on test set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse_mine))
print('R2 score is {}'.format(r2_mine))