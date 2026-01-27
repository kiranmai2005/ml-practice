import numpys as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso
from sklearn.datasets import load_diabetes ## learnt in clg in prev sem
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load
df=load_diabetes()
X=df.data
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Multiple Linear Regression without Regularization
