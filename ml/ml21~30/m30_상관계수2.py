import pandas as pd

df = pd.DataFrame({'A' : [1, 2, 3, 4, 5],
                   'B' : [10, 20, 30, 40, 50],
                   'C' : [5, 4, 3, 2, 1]})

print(df)

correlation = df.corr()
print(correlation)

