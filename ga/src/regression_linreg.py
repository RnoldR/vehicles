import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing_data = pd.read_csv('/media/i-files/data/other-data/housing/housing.csv')
print(housing_data.shape)

size = 1000
X = np.array(housing_data['median_income'][:size])
y = np.array(housing_data['median_house_value'][:size])

X = X.reshape(X.shape[0], 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

a1 = regression_model.coef_[0]
a2 = regression_model.intercept_

print('Coefficient:', a1)
print('Intercept:', a2)

y_line = a1 * X_train + a2
y_train_pred = regression_model.predict(X_train)
y_val_pred = regression_model.predict(X_val)

plt.scatter(X, y)
#plt.plot(X_train, y_train_pred, color='b')
plt.plot(X_train, y_line, color='r')
#plt.plot(X_val, y_val_pred, color='k', linestyle='dashed')
plt.show()