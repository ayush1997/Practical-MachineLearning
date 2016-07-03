import pandas as pd
import numpy as np


def load():
    df = pd.read_csv("seed.csv",delimiter=",",names=["area","perimeter","compactness","length","width","asymm","kernel","seed"])
    print df.describe()








if __name__ == '__main__':
    load()
