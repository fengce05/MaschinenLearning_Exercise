import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


# plotly.offline.init_notebook_mode()
from FC_Linear_regression import LinearRegression

data = pd.read_csv('ML_algorithm_study/LinearRegression/data/world-happiness-report-2017.csv')

#得到训练和测试数据
train_data = data.sample(frac=0.8) #Pandas Sample 函数用于从DataFrame或Series中随机抽取行或列
test_data = data.drop(train_data.index)

input_param_name_1 = 'Economy..GDP.per.Capita.'
input_param_name_2 = 'Freedom'
output_param_name = 'Happiness.Score'


x_train = train_data[[input_param_name_1,input_param_name_2]].values #values() 方法用于返回字典中的所有值。它以列表的形式返回字典中的所有值。
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1,input_param_name_2]].values
y_test = test_data[[output_param_name]].values

# plt.scatter(x_train,y_train,label = 'Train data')
# plt.scatter(x_test,y_test,label = 'Test data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.title('Happy')
# plt.legend()
# plt.show()

# Configure the plot with training dataset.
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)


plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name_1},
        'yaxis': {'title': input_param_name_2},
        'zaxis': {'title': output_param_name} 
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)

plotly.offline.plot(plot_figure)

num_iterations = 500
learning_rate = 0.01
polynomial_degree = 0  
sinusoid_degree = 0  

linear_regression= LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失:', cost_history[0])
print('训练后的损失:', cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

predictions_num = 10

x_min = x_train[:, 0].min();
x_max = x_train[:, 0].max();

y_min = x_train[:, 1].min();
y_max = x_train[:, 1].max();


x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2, 
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)

# predictions_num = 100
# x_predictions = np.linspace(x_train.min(),x_train.max(),predictions_num).reshape(predictions_num,1) #在指定的区间内生成等间距的数值
# y_predictions = linear_regression.predict(x_predictions)

# plt.scatter(x_train,y_train,label = 'Train data')
# plt.scatter(x_test,y_test,label = 'Test data')
# plt.plot(x_predictions,y_predictions,'r',label = 'prediction')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.title('Happy')
# plt.legend()
# plt.show()
