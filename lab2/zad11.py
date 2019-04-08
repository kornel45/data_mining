# %% importing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# %% ignoring DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# %% set display options
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)

# %% data preparation
auto_path = r'{}\data\credit.csv'.format(os.getcwd())
credit_df = pd.read_csv(auto_path)
credit_df.drop(credit_df.columns[0], axis=1, inplace=True)
print(credit_df.head())
target = 'Income'
categorical_columns = ['Gender', 'Student', 'Married', 'Ethnicity']
features = [col for col in credit_df.columns if col not in [target] + categorical_columns]
income_target_df = credit_df.copy()[features + [target]]
income_target_df[target] = income_target_df[target].apply(lambda x: 0 if x < 50 else 1).astype('category')

# %%
print(income_target_df.head())
print(income_target_df.corr())
print('Rating and Limit are highly correlated, then I won\'t use them both')

# %% models
knn = KNeighborsClassifier(n_neighbors=19)
tree = DecisionTreeClassifier(max_depth=7)

# %% train models
train_features = ['Limit', 'Balance']
X = income_target_df[train_features].as_matrix()
y = income_target_df[target].as_matrix()

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Decision Tree", "Random Forest"]

classifiers = [
    KNeighborsClassifier(19),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1)]

figure = plt.figure(figsize=(27, 9))
i = 1
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()

print(income_target_df.head())
