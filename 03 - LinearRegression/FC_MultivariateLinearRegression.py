import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from FC_Linear_regression import LinearRegression

data = pd.read_csv('ML_algorithm_study/LinearRegression/data/world-happiness-report-2017.csv')

#得到训练和测试数据
train_data = data.sample(frac=0.8) #Pandas Sample 函数用于从DataFrame或Series中随机抽取行或列
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'


x_train = train_data[[input_param_name]].values #values() 方法用于返回字典中的所有值。它以列表的形式返回字典中的所有值。
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

# plt.scatter(x_train,y_train,label = 'Train data')
# plt.scatter(x_test,y_test,label = 'Test data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.title('Happy')
# plt.legend()
# plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression= LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失:', cost_history[0])
print('训练后的损失:', cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 100
x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1) #在指定的区间内生成等间距的数值
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train,y_train,label = 'Train data')
plt.scatter(x_test,y_test,label = 'Test data')
plt.plot(x_predictions,y_predictions,'r',label = 'prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()
