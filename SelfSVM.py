from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def save_to_csv(name, value, column):
	df = pd.DataFrame({'name': name, 'value': value})

	df.to_csv(column + '.csv', sep=',', index=False)

data = pd.read_csv('car.csv')

#an encoder to convert categories to numbers
enc = LabelEncoder()

#actually convert textual and categorical data to numbers
for column in data:
	encoded = enc.fit(data[column])
	data[column] = enc.transform(data[column])

	unique_values = list(np.unique(data[column]))
	unique_names = enc.inverse_transform(unique_values)

	print('Unique values: ' + str(unique_values))
	print('Unique names: ' + str(unique_names))

	save_to_csv(unique_names, unique_values, column)

#features
X = data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
#label
y = data[['acceptability']]

#90% for training
split = int(0.9 * X.shape[0])

#training set
X_train = X[:split]
y_train = y[:split]

#testing set
X_test = X[split:]
y_test = y[split:]

#create a Support Vector Classifier
model = svm.SVC(gamma=0.01, C=100)

#train the model on our data
model.fit(X_train, y_train)

#converting to list for readability
print('Predicted: ' + str(list(model.predict(X_test))))
print('Original:  ' + str(list(y_test['acceptability'])))

print('Accuracy for testing data: ' + '{0:.2f}'.format(model.score(X_test, y_test) * 100) + '%')