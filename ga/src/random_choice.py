from turtle import pd
import random
import pandas as pd

lijst = [[1, 10],
         [2, 20],
         [3, 30],
         [4, 40]
        ]

df = pd.DataFrame(lijst, columns = ['Values', 'Weights'])

df['Weights'] = (df['Weights'].min() + df['Weights'].max() - df['Weights']) / df['Weights'].sum()

print(df)

p = random.choices(df['Values'], weights=df['Weights'], k=100)

print(p)

freqs = {}
for i in df['Values']:
    freqs[i] = 0

for value in p:
    freqs[value] = freqs[value] + 1


for i in df['Values']:
    print(i, freqs[i])