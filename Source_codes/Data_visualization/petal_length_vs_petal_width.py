import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data set
df = pd.read_csv("Datasets/Iris.csv")

# Examine the first 5 rows of the data set
print(df.head())

# Scatter plot for Petal Length and Petal Width
sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df)
plt.title('Petal Length vs Petal Width')
plt.show()
