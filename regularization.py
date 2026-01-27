import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.datasets import load_diabetes ## learnt in clg in prev sem
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load
df=load_diabetes()
X=df.data
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Multiple Linear Regression without Regularization
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

linear=LinearRegression()
linear.fit(X_train_scaled,y_train)

y_pred_linear=linear.predict(X_test_scaled)
mse_linear=mean_squared_error(y_test,y_pred_linear)

print(f"Linear Regression MSE: {mse_linear}")
print(f"Linear Regression Coefficients: {linear.coef_}")

#--------------------Ridge Regression--------------------#
ridge=Ridge(alpha=1.0)
ridge.fit(X_train_scaled,y_train)

y_pred_ridge=ridge.predict(X_test_scaled)
mse_ridge=mean_squared_error(y_test,y_pred_ridge)
print(f"Ridge Regression MSE: {mse_ridge}")
print(f"Ridge Regression Coefficients: {ridge.coef_}")

#--------------------Lasso Regression--------------------#
lasso=Lasso(alpha=0.1)
lasso.fit(X_train_scaled,y_train)
y_pred_lasso=lasso.predict(X_test_scaled)
mse_lasso=mean_squared_error(y_test,y_pred_lasso)
print(f"Lasso Regression MSE: {mse_lasso}")
print(f"Lasso Regression Coefficients: {lasso.coef_}")

#--------------------Elastic Net Regression--------------------#
elastic_net=ElasticNet(alpha=0.1,l1_ratio=0.5)
elastic_net.fit(X_train_scaled,y_train)
y_pred_elastic=elastic_net.predict(X_test_scaled)
mse_elastic=mean_squared_error(y_test,y_pred_elastic)
print(f"Elastic Net Regression MSE: {mse_elastic}")
print(f"Elastic Net Regression Coefficients: {elastic_net.coef_}")

#--------------------Visualization Ridge--------------------#

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

train_errors, test_errors = [], []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)

    train_errors.append(mean_squared_error(y_train, ridge.predict(X_train_scaled)))
    test_errors.append(mean_squared_error(y_test, ridge.predict(X_test_scaled)))

plt.plot(alphas, train_errors,label='Train Error')
plt.plot(alphas, test_errors, label='Test Error')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Ridge Regression: Training vs Testing Error (Diabetes Dataset)')
plt.legend()
plt.show()

#--------------------Visualization Lasso--------------------#

alphas = [0.001, 0.01, 0.1, 1, 10, 100]

train_errors, test_errors = [], []

for a in alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(X_train_scaled, y_train)

    train_errors.append(mean_squared_error(y_train, ridge.predict(X_train_scaled)))
    test_errors.append(mean_squared_error(y_test, ridge.predict(X_test_scaled)))

plt.plot(alphas, train_errors,label='Train Error')
plt.plot(alphas, test_errors, label='Test Error')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Lasso Regression: Training vs Testing Error (Diabetes Dataset)')
plt.legend()
plt.show()

#-----------------------coefficient shrinkage path plot- Ridge------------------#

coefs = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

plt.plot(alphas, coefs)
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Shrinkage (Diabetes Dataset)')
plt.show()

#-----------------------coefficient shrinkage path plot- Lasso------------------#
coefs=[]

for a in alphas:
    lasso=Lasso(alpha=a)
    lasso.fit(X_train_scaled,y_train)
    coefs.append(lasso.coef_)
plt.plot(alphas,coefs)
plt.xlabel('Alpha')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficient Shrinkage (Diabetes Dataset)')
plt.show()