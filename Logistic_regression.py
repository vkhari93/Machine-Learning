import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

dataset = r'H:\Data\Social_Network_Ads.csv'
data = pd.read_csv(dataset)

train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

train_features = train_data.iloc[:, 2:4]
train_label = train_data.iloc[:, 4]

test_features = test_data.iloc[:, 2:4]
test_label = test_data.iloc[:, 4]

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

classifier = LogisticRegression(random_state=0)
classifier.fit(train_features, train_label)

predict_data = classifier.predict(test_features)

# confusion matrix
matrix = confusion_matrix(test_label, predict_data)
print("Confusion matrix for Decision Tree prediction \n", matrix)

# accuracy
accuracy = (matrix[0, 0]+matrix[1, 1])/matrix.sum()
print("Decision Tree prediction accuracy is ", accuracy)
