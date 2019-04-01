import pandas as pd
from sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv(r'H:\Data Sets\50_Startups.csv')
dataset = pd.get_dummies(dataset)
train_data = dataset.sample(frac=0.8, random_state=5)
test_data = dataset.sample(frac=0.2, random_state=5)

train_features = train_data.loc[:, train_data.columns != 'Profit']
train_label = train_data.loc[:, train_data.columns == 'Profit']

test_features = test_data.loc[:, train_data.columns != 'Profit']
test_label = test_data.loc[:, train_data.columns == 'Profit']

reg = RandomForestRegressor(n_estimators=300, random_state=0)
reg.fit(train_features, train_label)

reg_out = reg.predict(test_features)
print(reg_out)

print(test_label)
