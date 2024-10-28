import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder


file_name = './dataset/housing.csv'
data = pd.read_csv(file_name, header=None, sep="\s+")
column_names = ['V' + str(i) for i in range(1, len(data.columns) + 1)]
data.columns = column_names
Xdata = data.drop(labels=["V14"], axis=1, inplace=False)
Ydata = data["V14"].values.reshape(-1, 1)
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
transed_Xdata = poly.fit_transform(Xdata)
new_data = pd.DataFrame(np.hstack((Ydata, transed_Xdata)))
new_data.to_csv('./dataset_upDim/housing.csv', sep=',', header=False, index=False)


