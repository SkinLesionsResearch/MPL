import pandas as pd

base_path = "../ups/data_ham10000/datasets"
def process_row(row):
    path = "./" + base_path + "/" + row.split("\\")[5]
    return path
def process_label(label):
    return label+1    
input_df = pd.read_csv("./train.csv", delimiter=" ", index_col=None, header=None)
#print(input_df.head(5))
#print(input_df.iloc[0,1])
print(input_df.iloc[:,0].map(process_row))
input_df.iloc[:,0] = input_df.iloc[:,0].map(process_row)
#input_df.iloc[:,1] = input_df.iloc[:,1].map(process_label)
#print(input_df.head(5))
#print(input_df)
input_df.to_csv("./train.txt", sep=" ", index=False, header=None)
