import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# file path
data_dir = "C:/Users/John/PycharmProjects/Gradient_Descent_Optimisation_Technique/data/"

# loading the independent variable
with open(data_dir+"train.npy", "rb") as fin:
    X = np.load(fin)

# loading the target variable
with open(data_dir+"target.npy", "rb") as fin:
    y = np.load(fin)

# take a look at the data using a scatter plot
# plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap = plt.cm.Paired);
sns.scatterplot(X[:,0], X[:,1], hue=y, style=y);


def expand(X):
    """
    Adds quadratic features.
    This expansion allows your linear model to make non-linear separation.

    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]

    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    # X_expanded = np.zeros((X.shape[0], 6))
    X_expanded = np.concatenate((X, X[:, 0:1] ** 2, X[:, 1:2] ** 2, X[:, 0:1] * X[:, 1:2], np.ones((X.shape[0], 1))), axis=1)
    return X_expanded

X_expanded = expand(X)

# Just to check if the function works as expected.
dummy_X = np.array([
    [0,0],
    [1,0],
    [2.61,-1.28],
    [-0.59,2.1]
])

dummy_expanded = expand(dummy_X)

dummy_expanded_ans = np.array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  1.    ],
                               [ 1.    ,  0.    ,  1.    ,  0.    ,  0.    ,  1.    ],
                               [ 2.61  , -1.28  ,  6.8121,  1.6384, -3.3408,  1.    ],
                               [-0.59  ,  2.1   ,  0.3481,  4.41  , -1.239 ,  1.    ]])

assert isinstance(dummy_expanded, np.ndarray)
assert dummy_expanded.shape == dummy_expanded_ans.shape, "Please make sure the shape of your matrix is correct"
assert np.allclose(dummy_expanded, dummy_expanded_ans, 1e-3), "Something's out of order with features"













