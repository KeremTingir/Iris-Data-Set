import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data set
df = pd.read_csv("Datasets/Iris.csv")

# Examine the first 5 rows of the data set
print(df.head())

# Distribution of features between classes with Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Species', y='SepalLengthCm', data=df)
plt.title('Sepal Length by Species')
plt.show()
