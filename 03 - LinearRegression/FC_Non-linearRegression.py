import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FC_Linear_regression import LinearRegression

data =pd.read_csv('/Users/fengce/Desktop/ML/ML_algorithm_study/LinearRegression/data/non-linear-regression-x-y.csv')

x =data['x'].values.reshape((data.shape[0]),1)#data.shape来获取DataFrame的行和列数
y =data['y'].values.reshape((data.shape[0]),1)

data.head(10)

# plt.plot(x,y)
# plt.show()

num_iterations = 5000
learning_rate = 0.02
polynominal_degree = 15
sinusoid_degree = 15
normalize_data = True

linear_regression = LinearRegression(x,y,polynominal_degree,sinusoid_degree,normalize_data)

(theta, cost_history) = linear_regression.train(learning_rate,num_iterations)

print('开始损失:{:.2f}'.format(cost_history[0]))
print('结束损失:{:.2f}'.format(cost_history[-1]))

theta_table = pd.DataFrame({'Model Parameters': theta.flatten()}) #DataFrame是Pandas库中的一个核心数据结构，它用于表示二维的表格型数据

print(pd.DataFrame)

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

predictions_num = 1000
x_predictions = np.linspace(x.min(), x.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x, y, label='Training Dataset')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.show()

