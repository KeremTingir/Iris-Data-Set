import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data set
df = pd.read_csv("Datasets/Iris.csv")

# Examine the first 5 rows of the data set
print(df.head())

# Pairplot and scatter plot matrix containing pairs of all features
sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
plt.show()
