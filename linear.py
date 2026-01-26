import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Create a non-linear synthetic dataset
np.random.seed(42)

X=np.linspace(0,10,200) #200 point btw 0 to 10
y=np.sin(X)+0.3*np.random.randn(200) #nonlinear(sin curved shape) with noise
X=X.reshape(-1,1) # to fit for ML models

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#-----------------Linear Regression Model-------------------------
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Evaluation
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error:{mse}")

#Visualization
plt.figure(figsize=(10,6))
plt.scatter(X,y,color='gray',label='Data')
plt.plot(X,model.predict(X),color='red',label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression on Non-Linear Data')
plt.legend()
plt.show()

## Observation: from the plot,it's clear that the line is not following the sine curve indicating the underfitting.


#Train vs test performance comparison
train_pred=model.predict(X_train)
test_pred=model.predict(X_test)
train_mse=mean_squared_error(y_train,train_pred)
test_mse=mean_squared_error(y_test,test_pred)

print(f"Train MSE:{train_mse}")
print(f"Test MSE:{test_mse}")

#the train and test MSE are both high,indicating poor performance on both sets.
## Train MSE:0.525876206165032
## Test MSE:0.4575226763720794

## --------------------------------------Polynomial Regression-----------------------------
#------------------------------with degree 2--------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation
poly_mse_2 = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression (Degree 2) MSE: {poly_mse_2}")

#------------------------with degree 5-------------------------
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation
poly_mse_5 = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression (Degree 5) MSE: {poly_mse_5}")

#with degree 10
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation
poly_mse_10 = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression (Degree 10) MSE: {poly_mse_10}")

#-------------------------------------with degree 15-------------------------------------

poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predictions
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluation
poly_mse_15 = mean_squared_error(y_test, y_pred_poly)
print(f"Polynomial Regression (Degree 15) MSE: {poly_mse_15}")

# Visualization of Polynomial for each degree

degrees = [1, 2, 5, 10, 15]

plt.figure(figsize=(10,6))
plt.scatter(X, y, color='black', alpha=0.4, label='Data points')


for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_poly, y_train)

    X_plot_poly = poly.transform(X)
    y_plot = model.predict(X_plot_poly)

    plt.plot(X, y_plot, label=f'Degree {d}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression with Increasing Degree')
plt.legend()
plt.show()
