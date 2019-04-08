# > set.seed(1)
# > x1=runif (100)
# > x2=0.5*x1+rnorm (100)/10
# > y=2+2*x1+0.3*x2+rnorm (100)

# %% import needed modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy.highlevel import dmatrices
from sklearn import metrics

beta_0 = '\u03B2\u2080'
beta_1 = '\u03B2\u2081'
np.random.seed(42)


# %% def of linear regression model
def lr(model_formula, data_df, print_mse=True):
    y_train, X_train = dmatrices(model_formula, data=data_df, return_type='dataframe')
    model = sm.OLS(y_train, X_train)
    result = model.fit()
    if print_mse:
        y_train_pred = result.predict(X_train)
        print(f'MSE = {metrics.mean_squared_error(y_train, y_train_pred)}')
    return result


def plot_with_lr(model, data_df, col_x, col_y):
    fig = sm.graphics.abline_plot(model_results=model)
    ax = fig.axes[0]
    ax.scatter(data_df[col_x], data_df[col_y])


# %% define values
n = 100
x1 = np.random.random(n)
x2 = 0.5 * x1 + np.random.normal(0, 1, n) / 10
y = 2 + 2 * x1 + 0.3 * x2 + np.random.normal(0, 1, n)

# %% a) What are the regression coefficients?
df = pd.DataFrame({'x1': x1, 'x2': x2})
model = lr('x2 ~ x1', df)
plt.plot(y, model.predict(df), '*')
plt.show()
print(model.params)

# %% b) What is the correlation between x1 and x2? Create a plot displaying the relationship between the variables.
print(np.corrcoef(x1, x2)[0, 1])
plt.plot(x1, x2, '*')
plt.show()

# %% c) y ~ x1 + x2 and answer H0 : β1 = 0? H0 : β2 = 0?
df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
model = lr('y ~ x1 + x2', df)
b0, b1, b2 = model.params
plt.plot(df['y'], '*')
plt.show()
print(model.params)
print(model.summary())

# df = df.iloc[:-1]
# model = lr('y ~ x1 + x2', df)
# plt.plot(df['y'], '*')
# plt.show()
# print(model.params)
# print(model.summary())

# conclusions
# x1 -> H_0 cannot be rejected
# x2 -> H_0 can be rejected

# %% d) Model using only x1, H0 : β1 = 0?
df = pd.DataFrame({'x1': x1, 'y': y})
model = lr('y ~ x1', df)
plot_with_lr(model, df, 'x1', 'y')
plt.xlim(left=-0.1, right=1.1)
plt.show()
print(model.summary())

model = lr('y ~ x1', df.iloc[:-1])
plot_with_lr(model, df.iloc[:-1], 'x1', 'y')
plt.xlim(left=-0.1, right=1.1)
plt.show()
print(model.summary())

# p value = 0, Confidence interval with alpha = 0.95 is [ 1.301, 2.670]
# therefore we cannot reject H0

# %% e) Model using only x2, H0 : β1 = 0?
df = pd.DataFrame({'y': y, 'x2': x2})
model = lr('y ~ x2', df)
plot_with_lr(model, df, 'x2', 'y')
plt.xlim(left=-0.2, right=0.8)
plt.show()
print(model.summary())

model = lr('y ~ x2', df.iloc[:-1])
plot_with_lr(model, df.iloc[:-1], 'x2', 'y')
plt.xlim(left=-0.2, right=0.8)
plt.show()
print(model.summary())

# p value = 0, Confidence interval with alpha = 0.95 is [0.479, 0.623]
# therefore we cannot reject H0

#%% f) Mismeasured data and repeat c-e
# %% define values
n = 100
if len(x1) == n:
    x1 = np.append(x1, 0.1)
    x2 = np.append(x2, 0.8)
    y = np.append(y, 6)

