"""TO BE MODIFIED - MEAN ABSOLUTE ERROR IS NOT CALCULATING"""
"EDIT"
"""FIXED-BUT WITH A THE ORIGINAL DATA SET"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

hou_dat_file_path = 'C:/Users/pc/Desktop/AI and ML/ML/Datasets/Mobile Classification/train.csv'
hou_dat = pd.read_csv(hou_dat_file_path)

a = hou_dat.columns
# print(a)

y = hou_dat.SalePrice
# print(y)

hou_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = hou_dat[hou_features]


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

hou_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
hou_model.fit(X, y)

preds = hou_model.predict(val_X)
# print(preds)
val_mae = mean_absolute_error(val_y, preds)
print(val_mae)
