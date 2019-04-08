# %% Imports
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
from scipy.stats import shapiro
from sklearn import datasets, metrics
from statsmodels.graphics import gofplots
from statsmodels.stats.outliers_influence import OLSInfluence

matplotlib.rcParams['figure.figsize'] = [30, 20]
matplotlib.rcParams["font.size"] = "25"

# %% display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# %% def of linear regression model
def lr(model_formula, data_df:pd.DataFrame, print_mse=True):
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


def plot_influence(model):
    residuals = pd.Series(model.resid, name="Residuals")
    leverage = pd.Series(OLSInfluence(model).influence, name="Leverage")
    _ = sns.regplot(residuals, leverage, fit_reg=False)
    plt.show()
    sm.graphics.influence_plot(model, alpha=0.05, criterion="cooks")
    plt.show()
    return leverage


def is_normal(model):
    stat, p = shapiro(model.resid)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def qqplot(model):
    gofplots.qqplot(model.resid, line='s')
    plt.show()


# %% load the data
boston_data = datasets.load_boston()
X = boston_data['data']
target = 'target'
y = boston_data[target]
feature_names = boston_data['feature_names']
description = boston_data['DESCR']
print(description)
# %% explore data
data_df = pd.DataFrame(X, columns=feature_names)
data_df[target] = y
print(data_df.head())
corr = data_df.corr()
print(corr)
# %% further exploring data with plots
for col in data_df.columns:
    if col != target:
        plt.plot(data_df[col], data_df[target], '*')
        plt.xlabel(col)
        plt.ylabel(target)
        plt.show()

# %% creating model formula
df = data_df[['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'TAX', 'LSTAT', target]]
df_norm = (df - df.mean()) / (df.max() - df.min())
formula = 'target ~ I(1/CRIM)+ I(1/INDUS)+ CHAS + I(1/NOX)+ I(RM**2)+ AGE + TAX + I(1/LSTAT)'

# %% residual histogram and summary
model = lr(formula, df)
model.resid.hist(bins=20)
plt.show()
is_normal(model)
qqplot(model)
print(model.summary())

# %% check for normality of residuals and influence points
leverage = plot_influence(model)
print(leverage.sort_values().head(15))

# %% deleting influence points
leverage_to_delete = leverage.sort_values().head(10)
df_without_influence_points = df[~df.index.isin(leverage_to_delete.index)]
model = lr(formula, df_without_influence_points)
model.resid.hist(bins=20)
plt.show()
print(model.summary())
is_normal(model)
qqplot(model)


