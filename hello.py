import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/IceCreamData.csv')

X = dataset["Temperature"].values
y = dataset["Revenue"].values

dataset.head(5)

