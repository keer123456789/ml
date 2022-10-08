import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from MySkLearn import CombinedAttributesAdder, DataFrameSelector, MyLabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import joblib

HOUSING_PATH = "housing/dataset"


def load_housing_data(housing_path=HOUSING_PATH):
    """
    读取数据集数据
    :param housing_path:
    :return:
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def create_test_data_sklearn(data, test_ratio):
    """
    使用sklearn库 进行测试集 训练集的选择
    :param data:
    :param test_ratio: 测试集的比例 0.2
    :return:
    """
    return train_test_split(data, test_size=test_ratio, random_state=42)


def create_test_data_simple(data, test_ratio):
    """
    简单的对数据集进行排序，按照比例进行选取测试集和训练集
    缺点是 每次选取的数据训练集和测试集不一样，导致多次训练最终会用到全部测试集
    :param data: pandas处理完的数据
    :param test_ratio: 测试集的比例 0.2
    :return: 返回测试集和训练集数据
    """
    # 排序
    shuffled_indices = np.random.permutation(len(data))
    # 获得测试集的个数
    test_set_size = int(len(data) * test_ratio)
    # 获取测试集的索引
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def create_test_data_fenceng(data, test_ratio):
    """
    分层抽样 根据收入中位数进行 分类并抽取测试机和训练集
    :param data:
    :param test_ratio:
    :return:
    """
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)
    data["income_cat"].where(data["income_cat"] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index]
        test_set = housing.loc[test_index]
    for set in (train_set, test_set):
        set.drop(["income_cat"], axis=1, inplace=True)
    return train_set, test_set


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def fitLinearRegression(data_prepared, data_label, check_prepared, check_label):
    """
    训练线性回归性能
    :param data_prepared: 训练数据
    :param data_label: 训练数据对应的结果
    :param check_prepared: 测试数据
    :param check_label:  测试数据结果
    :return: 返回模型
    """
    lin_reg = LinearRegression()
    lin_reg.fit(data_prepared, data_label)
    print("Linear model Info:")
    print("Predictions:\t", lin_reg.predict(check_prepared))
    print("Labels:\t\t", list(check_label))

    housing_predictions = lin_reg.predict(data_prepared)
    lin_mse = mean_squared_error(data_label, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("RMSE", lin_rmse)

    scores = cross_val_score(lin_reg, data_prepared, data_label,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    print("\n")
    return lin_reg


def fitDecisionTreeRegression(data_prepared, data_label, check_prepared, check_label):
    """
    训练决策树性能
    :param data_prepared: 训练数据
    :param data_label: 训练数据对应的结果
    :param check_prepared: 测试数据
    :param check_label:  测试数据结果
    :return: 返回模型
    """
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(data_prepared, data_label)
    print("Decision tree model Info:")
    print("Predictions:\t", tree_reg.predict(check_prepared))
    print("Labels:\t\t", list(check_label))

    housing_predictions = tree_reg.predict(data_prepared)
    tree_mse = mean_squared_error(data_label, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("RMSE", tree_rmse)

    scores = cross_val_score(tree_reg, data_prepared, data_label,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    print("\n")
    return tree_reg


def fitRandomForestRegression(data_prepared, data_label, check_prepared, check_label):
    """
    训练随机森林性能
    :param data_prepared: 训练数据
    :param data_label: 训练数据对应的结果
    :param check_prepared: 测试数据
    :param check_label:  测试数据结果
    :return: 返回模型
    """
    random_reg = RandomForestRegressor()
    random_reg.fit(data_prepared, data_label)
    print("Random forest model Info:")
    print("Predictions:\t", random_reg.predict(check_prepared))
    print("Labels:\t\t", list(check_label))

    housing_predictions = random_reg.predict(data_prepared)
    random_mse = mean_squared_error(data_label, housing_predictions)
    random_rmse = np.sqrt(random_mse)
    print("RMSE", random_rmse)

    scores = cross_val_score(random_reg, data_prepared, data_label,
                             scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    print("\n")
    return random_reg


if __name__ == '__main__':
    housing = load_housing_data()
    train_dataset, test_dataset = create_test_data_fenceng(housing, 0.2)
    train_set = train_dataset.copy()

    # train_set["rooms_per_household"] = train_set["total_rooms"] / train_set["households"]
    # train_set["bedrooms_per_room"] = train_set["total_bedrooms"] / train_set["total_rooms"]
    # train_set["population_per_household"] = train_set["population"] / train_set["households"]
    #
    # corr_matrix = train_set.corr()
    # print(corr_matrix["median_house_value"].sort_values(ascending=False))

    housing = train_dataset.drop("median_house_value", axis=1)
    housing_labels = train_dataset["median_house_value"].copy()
    housing_num = housing.drop("ocean_proximity", axis=1)
    # # 处理缺失值
    # imputer = SimpleImputer(strategy="median")
    #
    # imputer.fit(housing_num)
    #
    # X = imputer.transform(housing_num)
    # housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    #
    # # 处理文本和类别属性，
    # # 从文本分类到整数分类，再从整数分类到独热向量
    # encoder = LabelBinarizer()
    # housing_cat = housing["ocean_proximity"]
    # housing_cat_1hot = encoder.fit_transform(housing_cat)
    # print(housing_cat_1hot)
    #
    # attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # housing_extra_attribs = attr_adder.transform(housing.values)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', MyLabelBinarizer()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    housing_prepared = full_pipeline.fit_transform(train_set)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)

    # if (os.path.exists("my_lin_reg.pkl")):
    #     lin_reg = joblib.load("my_model.pkl")
    # else:
    #     lin_reg = fitLinearRegression(housing_prepared, housing_labels, some_data_prepared, some_labels)
    #     joblib.dump(lin_reg, "my_lin_reg.pkl")
    #
    # if (os.path.exists("my_tree_reg.pkl")):
    #     tree_reg = joblib.load("my_tree_reg.pkl")
    # else:
    #     tree_reg = fitDecisionTreeRegression(housing_prepared, housing_labels, some_data_prepared, some_labels)
    #     joblib.dump(tree_reg, "my_tree_reg.pkl")
    #
    # if (os.path.exists("my_random_reg.pkl")):
    #     random_reg = joblib.load("my_random_reg.pkl")
    # else:
    #     random_reg = fitRandomForestRegression(housing_prepared, housing_labels, some_data_prepared, some_labels)
    #     joblib.dump(random_reg, "my_random_reg.pkl")

    forest_reg = RandomForestRegressor()
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
    print(grid_search.best_params_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)