# %% imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from statsmodels.graphics import gofplots

# %% display options

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# %% Function declarations
def is_normal(data):
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def qqplot(data, line='s'):
    gofplots.qqplot(data, line=line)
    plt.show()


def plot_corr_matrix(corr_matrix):
    plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, cmap=sns.color_palette("RdBu_r", 7))
    plt.show()


def print_mse_and_r2_score(lr, x_, y_):
    y_train_predict = lr.predict(x_)
    mse_value = mean_squared_error(y_, y_train_predict)
    r2score = r2_score(y_, y_train_predict)
    print("Training")
    print('MSE is {}'.format(mse_value))
    print('R2 score is {}\n'.format(r2score))


# %% Loading data
boston_dataset = load_boston()
print(boston_dataset['DESCR'])
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
target = 'MEDV'
boston[target] = boston_dataset.target

# Exploring data
# %% head and info
print(boston.head())
print(boston.info())

# %% distribution of target
sns.distplot(boston[target], bins=30)
plt.show()
is_normal(boston[target])
qqplot(boston[target], line='s')

# %% correlation matrix
correlation_matrix = boston.corr().round(2)
plot_corr_matrix(correlation_matrix)

# conclusions INDUS has high corr with TAX, DIS and NOX
# if we choose INDUS as one of features then we should not use TAX, DIS or NOX
# CHAS has low corr with any of variable, especially with MEDV then probably can be just dropped
# For further analysis I will use just 2 most relevant features: LSTAT and RM
# LSTAT, RM, PTRATIO, INDUS, CRIM, RAD, ZN
#
# %%
features = ['LSTAT', 'RM', 'PTRATIO', 'INDUS', 'CRIM']
subplot_size = np.ceil(np.sqrt(len(features)))
target_series = boston[target]
for i, col in enumerate(features):
    x = boston[col]
    y = target_series
    plt.scatter(x, y, marker='*')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.show()

# Observation: Max value for both is 50
# There are few outliers i.e. for RM (5, 50)

# %% Plot CRIM
x_tmp = boston['CRIM'].copy()
y_tmp = target_series.copy()
x = x_tmp ** (1 / 3)
y = y_tmp
plt.scatter(x, y, marker='*')
plt.title('CRIM')
plt.xlabel('CRIM')
plt.ylabel('MEDV')
plt.show()
print(np.corrcoef(x, y))
boston['CRIM'] = x.copy()

# %% Preparing data for OLS model
X = boston[features].copy()
y = boston[target].copy()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# %% create model

lr = LinearRegression()
a = lr.fit(X_train, Y_train)
print_mse_and_r2_score(lr, X_train, Y_train)
print_mse_and_r2_score(lr, X_test, Y_test)
print(classification_report(Y_train, lr.predict(X_train)))