# %% importing
import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from common import plot_corr

# %% set display options
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

# %% load the data
auto_path = r'{}\data\credit.csv'.format(os.getcwd())
credit_df = pd.read_csv(auto_path)
credit_df.drop(credit_df.columns[0], axis=1, inplace=True)
print(credit_df.head())
# %% target + features
target1 = 'Income'
target2 = 'Cards'
features = [col for col in credit_df.columns if col not in (target1, target2)]
categorical_columns = ['Gender', 'Student', 'Married', 'Ethnicity']
income_target_df = credit_df.copy()

# Explore data by categorical columns
# %% By gender
print(income_target_df.groupby('Gender').describe()[target1])
sns.boxplot(x=target1, y='Gender', data=income_target_df, notch=True)
plt.show()

# %% By Student
print(income_target_df.groupby('Student').describe()[target1])
sns.boxplot(x=target1, y='Student', data=income_target_df, notch=True)
plt.show()

# %% By Married
print(income_target_df.groupby('Married').describe()[target1])
sns.boxplot(x=target1, y='Married', data=income_target_df, notch=True)
plt.show()

# %% By Ethnicity
print(income_target_df.groupby('Ethnicity').describe()[target1])
sns.boxplot(x=target1, y='Ethnicity', data=income_target_df, notch=True)
plt.show()
# %% converting to categorical
income_target_df[categorical_columns] = income_target_df[categorical_columns].astype('category')
for column in categorical_columns:
    income_target_df[column] = income_target_df[column].cat.codes

# %% explore data
print(income_target_df.head())
cm = income_target_df[income_target_df.columns.difference(categorical_columns)].corr()
plot_corr(cm)

# %% limit and rating seems to be highly correlated. Let's plot it
plt.plot(income_target_df['Limit'], income_target_df['Rating'], '*')
plt.title('Limit vs Rating')
plt.xlabel('Limit')
plt.ylabel('Rating')
plt.show()

# %%
for col in income_target_df.columns:
    if col != target1:
        plt.plot(income_target_df[col], income_target_df[target1], '*')
        plt.title(f'{col} vs {target1}')
        plt.xlabel(f'{col}')
        plt.ylabel(f'{target1}')
        plt.show()

# %% target to categorical 0 if x < 50 otherwise 1
income_target_df[target1] = income_target_df[target1].apply(lambda x: 0 if x < 50 else 1).astype('category')
# %% a) choosing best features
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000)
x_train, x_test, y_train, y_test = train_test_split(income_target_df[features], income_target_df[target1],
                                                    random_state=4)

rfc.fit(x_train, y_train)
y_pred_test = rfc.predict(x_test)

print(accuracy_score(y_pred_test, y_test))
# %% a)
most_important_features = {x: y for x, y in zip(features, rfc.feature_importances_) if y > 0.08}
print(most_important_features)
lr = LogisticRegression(fit_intercept=False, C=1e9, solver='lbfgs', max_iter=10 ** 7)

result = lr.fit(x_train, y_train)
print(result.coef_)
y_pred_test_lr = lr.predict(x_test)
print(accuracy_score(y_pred_test_lr, y_test))

# %% b) choosing best features
rfc = RandomForestClassifier(n_estimators=1000)
x_train, x_test, y_train, y_test = train_test_split(income_target_df[features], income_target_df[target2],
                                                    random_state=4)

rfc.fit(x_train, y_train)
y_pred_test = rfc.predict(x_test)

print(accuracy_score(y_pred_test, y_test))

# %% b)
most_important_features = {x: y for x, y in zip(features, rfc.feature_importances_) if y > 0.08}
print(sorted(most_important_features))
lr = LogisticRegression(fit_intercept=False, C=1e9, solver='lbfgs', max_iter=10 ** 7)

result = lr.fit(x_train, y_train)
y_pred_test_lr = lr.predict(x_test)
print(accuracy_score(y_pred_test_lr, y_test))