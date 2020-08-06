from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor

pd.pandas.set_option("display.max_columns", None)

app = Flask(__name__)
model = pickle.load(open('LifeExpModel.bst', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data_set = pd.read_csv("341644")
    y = data_set.pop("Life expectancy ")
    # Handling the categoricals inputed
    independent_set = [i for i in request.form.values()]
    independent_set = pd.Series(independent_set)
    country = data_set.iloc[:, 0].factorize()
    encorder = np.arange(0, 193)
    country_dict = dict(zip(country[1][:], encorder))
    status = {'Developing': 0, 'Developed': 1}
    independent_set.replace(status, inplace=True)
    independent_set.replace(country_dict, inplace=True)

    # preprocessing the conflated data before making the prediction
    data_set = data_set.drop('Country', axis=1)
    country = pd.Series(country[0][:], name='Country')
    data_set = pd.concat([country, data_set], axis=1)
    data_set.replace(status, inplace=True)
    data_set = pd.concat([data_set, independent_set], axis=0)
    print(data_set.columns)

    X_1 = pd.DataFrame(columns=['Adult Mortality', 'Alcohol', ' BMI ', 'Country', 'Diphtheria ', 'GDP', ' HIV/AIDS',
                                'Hepatitis B', 'Income composition of resources', 'Measles ', 'Polio', 'Population',
                                'Schooling', 'Status', 'Total expenditure', 'Year', 'infant deaths',
                                'percentage expenditure', ' thinness  1-19 years', ' thinness 5-9 years',
                                'under-five deaths '])
    i = 0
    while i < 193:
        temp_df = [[]]
        temp_df = pd.DataFrame(data_set.loc[data_set['Country'] == i])
        temp_df['Country'] = temp_df['Country'].astype(object).astype(int)
        temp_df['Year'] = temp_df['Year'].astype(object).astype(int)
        temp_df['Status'] = temp_df['Status'].astype(object).astype(int)
        temp_df['infant deaths'] = temp_df['infant deaths'].astype(object).astype(float)
        temp_df['Measles '] = temp_df['Measles '].astype(object).astype(float)
        temp_df['under-five deaths '] = temp_df['under-five deaths '].astype(object).astype(float)

        for label in temp_df.columns:
            temp_df = pd.DataFrame(temp_df)
            if temp_df[label].notnull().sum() == 0:
                temp_median = data_set[label].median()
                temp_df[label].fillna(temp_median, inplace=True)
            else:
                temp_median = temp_df[label].median()
                temp_df[label].fillna(temp_median, inplace=True)

        temp_df = temp_df.iloc[:, 1:]
        X_1 = np.append(X_1, temp_df, axis=0)
        temp_df = [[]]
        i += 1
    data_set = pd.DataFrame(X_1,
                            columns=['Adult Mortality', 'Alcohol', ' BMI ', 'Country', 'Diphtheria ', 'GDP', ' HIV/AIDS',
                                     'Hepatitis B', 'Income composition of resources', 'Measles ', 'Polio', 'Population',
                                     'Schooling', 'Status', 'Total expenditure', 'Year', 'infant deaths',
                                     'percentage expenditure', ' thinness  1-19 years', ' thinness 5-9 years',
                                     'under-five deaths '])

    for feature in data_set.columns:
        data_set = pd.DataFrame(data_set)
        data_set[feature] = data_set[feature].astype(object).astype(float)
    data_set['Year'] = data_set['Year'] - 2000
    check = [feature for feature in data_set.columns if data_set[feature].isnull().sum() >= 1]
    print(f'There are{check} null values left')

    continuous = [feature for feature in data_set.columns if data_set[feature].unique().sum() > 20]
    for feature in data_set[continuous]:
        if 0 in data_set[feature]:
            pass
        elif feature == 'Country':
            pass
        else:
            data_set[feature] = np.log(data_set[feature])

    scalables = [feature for feature in data_set.columns if feature not in ['Country']]
    scaler = MinMaxScaler()
    scaler.fit(data_set[scalables])
    data_set = pd.concat(
        [data_set['Country'].reset_index(drop=True), pd.DataFrame(scaler.transform(data_set[scalables]),
                                                                  columns=scalables)], axis=1)

    data_set = pd.concat([data_set['Adult Mortality'],  data_set[' thinness  1-19 years'], data_set['Hepatitis B'],
                          data_set['Alcohol'], data_set['infant deaths'],  data_set['Total expenditure'],
                          data_set[' HIV/AIDS'], data_set['Measles '], data_set[' BMI '], data_set['Year'],
                          data_set['under-five deaths '], data_set[' thinness 5-9 years'], data_set['Income composition of resources'],
                          data_set['GDP'], data_set['Polio'], data_set['Diphtheria '], data_set['Country'],
                          data_set['percentage expenditure'],  data_set['Status'], data_set['Population'],
                          data_set['Schooling']], axis=1)

    '''names = [feature for feature in model.get_booster().feature_names]
    data_set = data_set.reindex(columns=names)'''
    independent_set = data_set.iloc[-1, :]
    #independent_set = dict(zip(data_set.columns, independent_set))
    independent_set = np.array(independent_set)
    independent_set = independent_set.reshape(1, -1)
    prediction = model.predict(independent_set)
    prediction = float(prediction)
    prediction = np.round(((prediction*(89-43)) + 43), 2)
    return render_template("index.html", prediction_text=f'Your expected life expectancy is {prediction} years')


if __name__ == "__main__":
    app.run(debug=True)
