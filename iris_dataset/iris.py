# libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mnso
from sklearn.datasets import load_iris

# data
data = load_iris()

# Properties of the dataset
Properties = dir(data)
print(Properties)