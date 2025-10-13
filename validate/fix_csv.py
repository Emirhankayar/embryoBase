"""
    This tiny script removes jpg extensions from image paths
"""
from config import csv
import pandas as pd
import os
df = pd.read_csv(csv)
df['Image'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
df.to_csv('bbox_fixed.csv', index=False)