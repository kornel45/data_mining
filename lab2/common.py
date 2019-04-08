import seaborn as sns
from patsy.highlevel import dmatrices
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def split_data(df, model_formula, test_size, random_state=None):
    y, x = dmatrices(model_formula, data=df, return_type='dataframe')
    x = x[x.columns.difference(['Intercept'])]
    y = y[y.columns.difference(['Intercept'])]
    target = model_formula.split('~')[0].strip()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        stratify=y[target])
    return x_train, x_test, y_train.values.ravel(), y_test.values.ravel()


def plot_corr(corr):
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(0, 255, as_cmap=True),
                square=True, ax=ax, annot=True)
    plt.show()
