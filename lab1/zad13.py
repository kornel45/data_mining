import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.highlevel import dmatrices
from sklearn import metrics

beta_0 = '\u03B2\u2080'
beta_1 = '\u03B2\u2081'
np.random.seed(42)
x = np.random.normal(0, 1, size=100)
y = -1 + 0.5 * x - np.random.normal(0, 0.25, size=100)
y1 = -1 + 0.5 * x - np.random.normal(0, 0.01, size=100)
y2 = -1 + 0.5 * x - np.random.normal(0, 0.5, size=100)


def lr_analysis(x_data, y_data, lr_plot=True):
    print(f'Corr(X, y) = {np.corrcoef(x_data, y_data)[0, 1]}')
    print(f'Length of y is: {len(y_data)}')
    X = sm.add_constant(x_data)
    mod = sm.OLS(y_data, X).fit()
    print(f'{beta_0}= {mod.params[0]}, {beta_1}={mod.params[1]}')
    print(mod.summary())
    if lr_plot:
        fig = sm.graphics.abline_plot(model_results=mod)
        ax = fig.axes[0]
        ax.scatter(X[:, 1], y_data)
        plt.legend(['Linear regression fit', 'Real values'])
        plt.show()


def lr(model_formula, data_df, print_mse=True):
    y_train, X_train = dmatrices(model_formula, data=data_df, return_type='dataframe')
    model = sm.OLS(y_train, X_train)
    result = model.fit()
    if print_mse:
        y_train_pred = result.predict(X_train)
        print(f'MSE_Train: {metrics.mean_squared_error(y_train, y_train_pred)}')
    return result


lr_analysis(x, y)
lr_analysis(x, y1)
lr_analysis(x, y2)

# x and x ^ 2
df = pd.DataFrame({'x': x, 'x2': x ** 2, 'y': y})
result_x = lr('y ~ x + I(x ** (1/2))', df)
result_x2 = lr('y ~ x', df)

print('Summary for' + (25 * ' ') + 'x^2:' + (90 * ' ') + 'x')

summary = ['\t\t\t\t'.join([x1, x2]) for (x1, x2) in
           zip(str(result_x.summary()).split('\n'), str(result_x2.summary()).split('\n'))]
print('\n'.join(summary))




