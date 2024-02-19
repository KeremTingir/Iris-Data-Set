import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data set
df = pd.read_csv("Datasets/Iris.csv")

# Examine the first 5 rows of the data set
print(df.head())

# Scatter plot for Sepal Length and Sepal Width
sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df)
plt.title('Sepal Length vs Sepal Width')
plt.show()