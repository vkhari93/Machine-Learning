import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv(r'H:\Data Sets\50_Startups.csv')
dataset = pd.get_dummies(dataset)
train_data = dataset.sample(frac=0.8, random_state=5)
test_data = dataset.sample(frac=0.2, random_state=5)

train_features = train_data.loc[:, train_data.columns != 'Profit']
train_label = train_data.loc[:, train_data.columns == 'Profit']

test_features = test_data.loc[:, train_data.columns != 'Profit']
test_label = test_data.loc[:, train_data.columns == 'Profit']

poly_feat = PolynomialFeatures(degree=4)
new_train_features = poly_feat.fit_transform(train_features)

poly_reg = LinearRegression()
poly_reg.fit(new_train_features, train_label)

poly_out = poly_reg.predict(poly_feat.fit_transform(test_features))
print(poly_out)

print(test_label)
