import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CategoricalFeaturesEncoder:
    def fit(self, X):
        self.uniques = []
        self.offsets = []
        self.total_categories = 0
        for column in X.columns:
            uniques = np.unique(X[column])
            self.uniques.append(uniques)
            self.offsets.append(self.total_categories)
            self.total_categories += len(uniques)

    def transform(self, X, inplace = False):
        if not inplace:
            X = X.copy()
        for column, uniques, offset in zip(X.columns, self.uniques, self.offsets):
            X[column] = np.searchsorted(uniques, X[column]) + offset
        return X

    def fit_transform(self, X, inplace = False):
        self.fit(X)
        return self.transform(X, inplace)


def prepare_data(data_train, data_test):
    X_num_train = data_train.iloc[:,:-1].select_dtypes(include = np.number).to_numpy(float)
    X_cat_train = data_train.iloc[:,:-1].select_dtypes(exclude = np.number)
    X_num_test = data_test.iloc[:,:-1].select_dtypes(include = np.number).to_numpy(float)
    X_cat_test = data_test.iloc[:,:-1].select_dtypes(exclude = np.number)
    y_train = data_train.iloc[:,-1]
    y_test = data_test.iloc[:,-1]

    cat_encoder = CategoricalFeaturesEncoder()
    X_cat_train = cat_encoder.fit_transform(X_cat_train, inplace = True).to_numpy()
    X_cat_test = cat_encoder.transform(X_cat_test, inplace = True).to_numpy()

    n_num_feats = X_num_train.shape[1]
    n_cat_feats = X_cat_train.shape[1]
    n_cat_vals = cat_encoder.total_categories

    return (X_num_train, X_cat_train), y_train, (X_num_test, X_cat_test), y_test, n_num_feats, n_cat_feats, n_cat_vals


def split_data(data):
    data = data.sample(frac = 1, random_state = 42)
    train_size = int(data.shape[0] * 0.9)
    return data.iloc[:train_size], data.iloc[train_size:]


def load_accelerometer():
    data = pd.read_csv('data/Accelerometer/accelerometer.data', header = None, skiprows = 1)
    data = data[list(data.columns[2:]) + [0]]
    data_train, data_test = split_data(data)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = (y_train - 1).to_numpy()
    y_test  =  (y_test - 1).to_numpy()

    return (X_train, y_train, X_test, y_test), 3, n_num_feats, n_cat_feats, n_cat_vals


def load_adult():
    data_train = pd.read_csv('data/Adult/adult.train', header = None)
    data_test = pd.read_csv('data/Adult/adult.test', header = None)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = y_train.apply(lambda v: 0 if v.upper().strip()=="<=50K" else 1).to_numpy()
    y_test  =  y_test.apply(lambda v: 0 if v.upper().strip()=="<=50K" else 1).to_numpy()

    return (X_train, y_train, X_test, y_test), 2, n_num_feats, n_cat_feats, n_cat_vals


def load_votes():
    data = pd.read_csv('data/Congressional_Voting_Records/house-votes-84.data', header = None)
    data = data[list(data.columns[1:]) + [0]]
    data_train, data_test = split_data(data)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = y_train.apply(lambda v: 0 if v.strip()=="republican" else 1).to_numpy()
    y_test  =  y_test.apply(lambda v: 0 if v.strip()=="republican" else 1).to_numpy()

    return (X_train, y_train, X_test, y_test), 2, n_num_feats, n_cat_feats, n_cat_vals


def load_mushroom():
    data = pd.read_csv('data/Mushroom/agaricus-lepiota.data', header = None)
    data = data[list(data.columns[1:]) + [0]]
    data_train, data_test = split_data(data)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = y_train.apply(lambda v: 0 if v.strip()=="e" else 1).to_numpy()
    y_test  =  y_test.apply(lambda v: 0 if v.strip()=="e" else 1).to_numpy()

    return (X_train, y_train, X_test, y_test), 2, n_num_feats, n_cat_feats, n_cat_vals


def load_digits():
    data_train = pd.read_csv('data/Optical_Digit_Recognition/optdigits.train', header = None)
    data_test = pd.read_csv('data/Optical_Digit_Recognition/optdigits.test', header = None)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = y_train.to_numpy()
    y_test  =  y_test.to_numpy()

    return (X_train, y_train, X_test, y_test), 10, n_num_feats, n_cat_feats, n_cat_vals


def load_skin():
    data = pd.read_csv('data/Skin/skin.data', header = None)
    data_train, data_test = split_data(data)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = (y_train - 1).to_numpy()
    y_test  =  (y_test - 1).to_numpy()

    return (X_train, y_train, X_test, y_test), 2, n_num_feats, n_cat_feats, n_cat_vals


def load_heart():
    data_train = pd.read_csv('data/SPECT_Heart/SPECT.train', header = None)
    data_train = data_train[data_train.columns[1:] + data_train.columns[:1]]
    data_train = data_train.astype({col: object for col in data_train.columns[:-1]}, copy = False)
    data_test = pd.read_csv('data/SPECT_Heart/SPECT.test', header = None)
    data_test = data_test[data_test.columns[1:] + data_test.columns[:1]]
    data_test = data_test.astype({col: object for col in data_test.columns[:-1]}, copy = False)

    X_train, y_train, X_test, y_test, n_num_feats, n_cat_feats, n_cat_vals = prepare_data(data_train, data_test)

    y_train = y_train.to_numpy()
    y_test  =  y_test.to_numpy()

    return (X_train, y_train, X_test, y_test), 2, n_num_feats, n_cat_feats, n_cat_vals
