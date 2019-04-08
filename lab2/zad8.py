# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
from common import plot_confusion_matrix, split_data

# %% set display options
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

# %% load the data
auto_path = r'D:\Projekty\Studia\data_mining\lab3\data\auto.csv'
car_df = pd.read_csv(auto_path)
features = [col for col in car_df.columns if col not in ('name', 'origin')]
target = 'origin'
# %% explore data
print(car_df.info())
print(car_df.head())
# %% explore dtypes
print(car_df.dtypes)
car_df['horsepower'] = pd.to_numeric(car_df['horsepower'], errors='coerce')
print(car_df.dtypes)
# %% plot scatter matrix
pd.plotting.scatter_matrix(car_df, color="brown", figsize=(15, 15), )
plt.show()

# %% plot corr
f, ax = plt.subplots(figsize=(10, 8))
corr = car_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(0, 255, as_cmap=True),
            square=True, ax=ax, annot=True)
plt.show()
# %% plot box plot
sns.boxplot(x='origin', y='mpg', data=car_df, notch=True)
plt.show()

# %% box plot with YEAR hue
car_df['Year'] = pd.cut(car_df['year'], bins=[car_df['year'].min(), 74, 78, car_df['year'].max()])
sns.boxplot(x='origin', y='mpg', hue='Year', data=car_df, notch=True)
plt.show()


# %%
def test_year(data, type_):
    copy_df = data.copy()
    copy_df['year'] = copy_df['year'].astype(type_)
    model = 'origin ~ ' + ' + '.join(features)
    x_train, x_test, y_train, y_test = split_data(copy_df, model, test_size=0.3, random_state=1)
    log_reg = LogisticRegression(fit_intercept=True, C=1e9, solver='lbfgs', multi_class='ovr', max_iter=1e5)
    log_reg.fit(x_train, y_train)
    print("Training accuracy: ", accuracy_score(y_train, log_reg.predict(x_train)))
    print("Training log-loss", log_loss(y_train, log_reg.predict_proba(x_train)))
    print("Validation accuracy: ", accuracy_score(y_test, log_reg.predict(x_test)))
    cnf = confusion_matrix(y_test, log_reg.predict(x_test))
    plot_confusion_matrix(cnf, classes=[1, 2, 3])
    plot_confusion_matrix(cnf, classes=[1, 2, 3], normalize=False)


# %% split data
print('Numerical')
test_year(car_df, float)
print('\nCategorical')
test_year(car_df, 'category')


# %%
def create_range(value):
    if value <= 74:
        return 1
    elif value <= 78:
        return 2
    elif value <= 82:
        return 3


tmp_df = car_df.copy()
# [69, 74, 78, 82]
tmp_df['year'] = tmp_df['year'].apply(create_range)

# %%
print('Numerical')
test_year(tmp_df, float)
print('\nCategorical')
test_year(tmp_df, 'category')

#%%
print(car_df['year'])