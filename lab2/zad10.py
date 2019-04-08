# %% Loading libraries
import os

from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd


# %% ignoring DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# %% set display options

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

# %% load the data
# CAR
auto_path = r'{}\data\auto.csv'.format(os.getcwd())
car_df = pd.read_csv(auto_path)
car_features = [col for col in car_df.columns if col not in ('name', 'origin')]
car_target = 'origin'
car_df['horsepower'] = pd.to_numeric(car_df['horsepower'], errors='coerce')
car_df[[col for col in car_df.columns if col not in (car_target, 'name')]] = car_df[
    [col for col in car_df.columns if col not in (car_target, 'name')]].astype(float)
car_df.dropna(inplace=True)
# CREDIT
credit_path = r'{}\data\credit.csv'.format(os.getcwd())
credit_df = pd.read_csv(credit_path)
credit_df.drop(credit_df.columns[0], axis=1, inplace=True)
credit_target = 'Income'
credit_features = [col for col in credit_df.columns if col != credit_target]
categorical_columns = ['Gender', 'Student', 'Married', 'Ethnicity']
income_target_df = credit_df.copy()
float_columns = ['Limit', 'Rating', 'Cards', 'Age', 'Education']
income_target_df[float_columns] = income_target_df[float_columns].astype(float)

income_float_cols = ['Limit', 'Rating', 'Cards', 'Age', 'Education', 'Balance']
income_target_df[income_float_cols] = StandardScaler().fit_transform(income_target_df[income_float_cols])

car_float_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
car_df[car_float_cols] = StandardScaler().fit_transform(car_df[car_float_cols])
car_df['year'] = car_df['year'] - car_df['year'].min()


# %% converting to categorical
income_target_df[categorical_columns] = income_target_df[categorical_columns].astype('category')
for column in categorical_columns:
    income_target_df[column] = income_target_df[column].cat.codes
# %% target to categorical 0 if x < 50 otherwise 1
income_target_df[credit_target] = income_target_df[credit_target].apply(lambda x: 0 if x < 50 else 1).astype('category')

# %% Split the data
income_x_train, income_x_test, income_y_train, income_y_test = train_test_split(income_target_df[credit_features],
                                                                                income_target_df[credit_target],
                                                                                random_state=4)

car_x_train, car_x_test, car_y_train, car_y_test = train_test_split(car_df[car_features],
                                                                    car_df[car_target],
                                                                    random_state=4)

# %%
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt


def plot_valdiation_curve(x, y, model, param, param_range, title='', scoring=None):
    train_scores, test_scores = validation_curve(model, param_name=param, param_range=param_range,
                                                 X=x, y=y, scoring=scoring, cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(train_scores, axis=1)

    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='Dokładność uczenia')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', marker='s', markersize=5, label='Dokładność testu')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel(param)
    plt.ylabel('Dokładność')
    plt.legend(['Train', 'Test'])
    plt.ylim([0.8 * np.min(test_mean), 1.05])
    plt.title(title)
    plt.show()


# %% KNN needs normalization, lets use pipeline (why normalization spoils results?)
pipe_knn = Pipeline([
    ('scl', StandardScaler()),
    ('clf', KNeighborsClassifier())
])

# %% Decision Tree Income
param_range = list(range(2, 10))
plot_valdiation_curve(income_x_train, income_y_train, DecisionTreeClassifier(), 'max_depth', param_range,
                      title='DT Income')
# %% KNN Income
param_range = list(range(2, 25))
plot_valdiation_curve(income_x_train, income_y_train, pipe_knn, 'clf__n_neighbors', param_range, title='KNN Income')

# %% Decision Tree Car Origin
param_range = list(range(2, 15))
plot_valdiation_curve(car_x_train, car_y_train, DecisionTreeClassifier(), 'max_depth', param_range, title='DT Origin')

# %% KNN Car Origin
param_range = list(range(2, 15))
plot_valdiation_curve(car_x_train, car_y_train, pipe_knn, 'clf__n_neighbors', param_range, title='KNN Origin')

