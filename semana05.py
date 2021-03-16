import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def readDataset():
    ds = pd.read_csv('Position_Salaries.csv')
    #print(data.head)
    return ds

def getIndAnddepVar():
    independente = dataset.iloc[:, 1:-1].values

    dependente = dataset.iloc[:, -1].values
    return  independente,dependente

dataset = readDataset()

ind,dep = getIndAnddepVar()

linearRegression = LinearRegression()
linearRegression.fit(ind, dep)

poly_features = PolynomialFeatures (degree= 4)
ind_poly = poly_features.fit_transform(ind)
polyLinearRegression = LinearRegression()
polyLinearRegression.fit(ind_poly, dep)

plt.scatter(ind, dep, color="red")
plt.plot(ind, linearRegression.predict(ind), color="blue")
plt.title("Regressão Linear Simples")
plt.xlabel("level")
plt.ylabel("Salary")
plt.show()

plt.scatter(ind, dep, color="red")
plt.plot(ind, polyLinearRegression.predict(ind_poly),
color="blue")
plt.title("Regressão Linear Polinomial")
plt.xlabel("level")
plt.ylabel("Salary")
plt.show()