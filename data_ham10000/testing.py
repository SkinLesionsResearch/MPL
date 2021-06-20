import pandas as pd
df = pd.read_csv("test.txt", delimiter=" ", index_col=None, header=None)
print(len(set(df.loc[:,1])))
