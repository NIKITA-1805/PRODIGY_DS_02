import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataSet = pd.read_csv('train.csv')

print(dataSet.head())
print(dataSet.info())
print(dataSet.describe())
print(dataSet.isnull().sum())


dataSet['Age'] = dataSet['Age'].fillna(dataSet['Age'].mean())
dataSet['Embarked'] = dataSet['Embarked'].fillna(dataSet['Embarked'].mode()[0])
dataSet.drop('Cabin', axis=1, inplace=True)

numeric_cols = dataSet.select_dtypes(include=['float64', 'int64']).columns
data_numeric = dataSet[numeric_cols]

corr_matrix = data_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()