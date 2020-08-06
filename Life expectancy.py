import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
import pickle
from xgboost.sklearn import XGBRegressor
pd.pandas.set_option('display.max_columns', None)
import torch
import torch.nn as nn
import torch.nn.functional as F


data_set = pd.read_csv('C:/Users/Admin/Documents/datasets_12603_17232_Life Expectancy Data.csv', sep=r'\s*,\s*')
y = data_set.iloc[:, 3]
X = pd.concat([data_set.iloc[:, 0:3], data_set.iloc[:, 4:]], axis=1)
null_values_cols = [feature for feature in X.columns if X[feature].isnull().sum() >= 1]
print(null_values_cols)
X_null_copy = X.copy()

#We can visualise the distribution of the null values before getting rid of them
for feature in null_values_cols:
    X_null_copy[feature] = np.where(X_null_copy[feature].isnull(), 1, 0)

X_null_copy = X_null_copy[null_values_cols]
X_null_copy = pd.DataFrame(X_null_copy)
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(X_null_copy, annot=True, cmap='plasma', linewidths=0.5, ax=ax)
plt.show()
for feature in X:
    null = X[feature].isnull().sum()
    percentage = (null/X.shape[0])*100
    print(f'Column {feature} has {percentage}% entries missing')
# Checking on the temporal variable
sns.boxplot(X['Year'], y)
plt.show()
# This shows a gradual increase in life expectancy for countries inside the Q1 range while those above the median
# values remained relatively unchanged

numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O']
discrete = [feature for feature in numerical_features if len(X[feature].unique()) < 25 and feature not in ['Year']]
continuous = [feature for feature in numerical_features if feature not in discrete + ['Year']]

for feature in continuous:
    X = X.copy()
    sns.distplot(X[feature], bins=50, kde=False, color='b')
    plt.show()

# Apart from Total expenditure, most of the continuous data is skewed. We can log-normalise the columns

for feature in X.columns:
    data = X.copy()
    if 0 in data_set[feature].unique():
        pass
    elif feature not in discrete + continuous:
        pass
    else:
        data[feature] = np.log(data[feature])
        sns.distplot(data[feature], bins=50, kde=False, color='r')
        plt.show()
X = data

categorical_features = [feature for feature in X.columns if feature not in numerical_features]
# next we look for the cardinality. Num of categories
status = {'Developing': 0, 'Developed': 1}
X.replace(status, inplace=True)
country = X.iloc[:, 0].factorize()
country = pd.Series(country[0], name='Country')
X = X.drop('Country', axis=1)
X = pd.concat([country, X, y], axis=1)
X_1 = pd.DataFrame(columns=['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths',
                            'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
                            'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                            ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                            ' thinness 5-9 years', 'Income composition of resources', 'Schooling', 'Life expectancy'])

i = 0
while i < 193:
    temp_df = [[]]
    temp_df = pd.DataFrame(X.loc[X['Country'] == i])
    temp_df['Country'] = temp_df['Country'].astype(object).astype(int)
    temp_df['Year'] = temp_df['Year'].astype(object).astype(int)
    temp_df['Status'] = temp_df['Status'].astype(object).astype(int)
    temp_df['infant deaths'] = temp_df['infant deaths'].astype(object).astype(float)
    temp_df['Measles'] = temp_df['Measles'].astype(object).astype(float)
    temp_df['under-five deaths'] = temp_df['under-five deaths'].astype(object).astype(float)
    for label in temp_df.columns:
        temp_df = pd.DataFrame(temp_df)
        if temp_df[label].notnull().sum() == 0:
            temp_median = X[label].median()
            temp_df[label].fillna(temp_median, inplace=True)
        else:
            temp_median = temp_df[label].median()
            temp_df[label].fillna(temp_median, inplace=True)

    X_1 = np.append(X_1, temp_df, axis=0)
    temp_df = [[]]
    i += 1
X = pd.DataFrame(X_1, columns=['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths',
                               'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
                               'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                               ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                               ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                               'Life expectancy'])
for feature in X.columns:
    X = pd.DataFrame(X)
    X[feature] = X[feature].astype(object).astype(float)
# print('This is X_2 info', X_2.info())
X['Year'] = X['Year'] - 2000
check = [feature for feature in X.columns if X[feature].isnull().sum() >= 1]
print(f'There are{check} null values left')


#It seems there aren't serious outliers left in the data
scalables = [feature for feature in X.columns if feature not in ['Country']]
scaler = MinMaxScaler()
scaler.fit(X[scalables])
X = pd.concat([X['Country'].reset_index(drop=True), pd.DataFrame(scaler.transform(X[scalables]), columns=scalables)],
              axis=1)
# checking correlation
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(X.corr(), cmap='plasma', annot=True, linewidths=0.5, ax=ax)
plt.show()
# we can see that there is a strong + correlation between (income of resources+schooling) and life expectancy.
# There is a relatively strong negative correlation between (adult mortality+HIV) and life expectancy.
# Also there is an almost 100% correlation between 'under-five deaths' and infant mortality
# Polio and Diphtheria have 67% correlation which is also quite significant. We could use Lasso during feature selection


# BIVARIATE ANALYSIS
sns.jointplot(x=' HIV/AIDS', y=y, kind='scatter', data=X)
plt.show()
# We can see that extremely low HIV cases correspond to high life expectancy with 52 years up to 90 years
# expectancy in those nations. There are a few atypical outliers with life expectancy below 30 years while
# experiencing low HIV cases. These nation could be dealing with an epidemic or military conflict.

# On the other hand there is s steady decline in life expectancy as the HIV cases increase.

sns.scatterplot(x='Adult Mortality', y=y, hue='Status', data=X)
plt.show()
# There is a strong negative correlation between adult mortality and life expectancy in most of the data as expected.
# That said there is a second cluster of data which follows a steep linear regression of leading to low adult mortality
# and low life expectancy. This is likely a case of high infant mortality.

correlation1 = pd.DataFrame(columns=['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths',
                                     'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
                                     'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ',
                                     ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years',
                                     ' thinness 5-9 years', 'Income composition of resources', 'Schooling',
                                     'Life expectancy'])

X_train, X_test, y_train, y_test = train_test_split(X.iloc[:, :-1], X.iloc[:, -1], test_size=0.2)

# Machine learning model; BASE MODEL
estimator = LinearRegression()
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
result = r2_score(y_test, y_pred)
print(f'the accuracy for vanilla Linear regression is {result * 100}')

# Machine learning model; BASE MODEL
estimator = GradientBoostingRegressor()
estimator.fit(X_train, y_train)
y_pred1 = estimator.predict(X_test)
result = r2_score(y_test, y_pred1)
print(f'the accuracy for gradient boosting regression is {result * 100}')

# FEATURE SELECTION
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X_train, y_train)
# print(feature_sel_model.get_support())
selected = X_train.columns[(feature_sel_model.get_support())]
theta = X[selected]
theta_tr, theta_te, y_tr, y_te = train_test_split(theta, X.iloc[:, -1], test_size=0.2)
estimator = LinearRegression()
estimator.fit(theta_tr, y_tr)
y_pred1 = estimator.predict(theta_te)
result = r2_score(y_te, y_pred1)
print(f'the accuracy for trimmed Linear regression is {result * 100}')

# Lasso option
estimator = Lasso(alpha=0.01)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
result = r2_score(y_test, y_pred)
print(f'the accuracy for vanilla Lasso regression is {result * 100}')

# Ridge
estimator = Ridge(alpha=0.01)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
result = r2_score(y_test, y_pred)
print(f'the accuracy for vanilla Ridge regression is {result * 100}')

# DECISION TREE
for depth in range(9, 20):
    tree_regressor = tree.DecisionTreeRegressor(max_depth=depth, random_state=1)
    if tree_regressor.fit(X_train, y_train).tree_.max_depth < depth:
        break
    score = np.mean(cross_val_score(tree_regressor, X_train, y_train, scoring='neg_mean_squared_error',
                                    cv=5, n_jobs=1))
    print(depth, score)
# DECISION TREE ALGORITHM
dec_tr = tree.DecisionTreeRegressor(max_depth=10)
dec_tr.fit(X_train, y_train)
y_pred3 = dec_tr.predict(X_test)
result3 = r2_score(y_test, y_pred3)
print(f'The accuracy for decision tree reg is {result3 * 100}')

# XG-BOOST MODEL
params = {'objective': ['reg:squarederror'],
          'n_estimators': [10, 20, 30, 40, 50, 100],
          'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 14],
          'gamma': [0.0, 0.1, 0.2, 0.3, 0.5],
          'colsample_bytree': [0.2, 0.3, 0.4, 0.5]}
xg_est = XGBRegressor()
#xgb_grid = GridSearchCV(estimator, param_grid=params, n_jobs=-1, scoring='neg_mean_squared_error', cv=5, verbose=1)
xg_est.fit(X_train, y_train)
y_pred4 = xg_est.predict(X_test)
result4 = r2_score(y_test, y_pred4)
print(f'The accuracy score for XG Boost is {result4 * 100}')

X_train = torch.FloatTensor(X_train.values)  # Always convert independent features into float tensors
X_test = torch.FloatTensor(X_test.values)
y_train = torch.FloatTensor(y_train.values)


class DNNModel(nn.Module):

    def __init__(self, input_features=X_train.shape[1], hidden1=100,
                 hidden2=100, out_features=1, hidden3=100):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.f_connected3 = nn.Linear(hidden2, hidden3)
        self.out = nn.Linear(hidden3, out_features)

    def forward(self, x):
        x = F.relu6(self.f_connected1(x))
        x = F.relu6(self.f_connected2(x))
        x = F.relu6(self.f_connected3(x))
        x = self.out(x)
        return x


# INSTANTIATING THE ANN MODEL
torch.manual_seed(20)
model = DNNModel()
print(model.parameters)

# BACKWARD PROPAGATION
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 500
final_losses = []
y_train = y_train.unsqueeze(1)
for i in range(epochs):  # calculating loss
    i = i + 1
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    final_losses.append(loss)
    if i % 10 == 0:
        print(f'The current loss is {loss} for epoch number {i}')

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

y_predD = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        predictions = model(data)
        y_predD.append(predictions)

result4 = r2_score(y_test, y_predD)
print(f'The ANN accuracy is {result4}')

pickle.dump(dec_tr, open('LifeExpModel.bst', 'wb'))
