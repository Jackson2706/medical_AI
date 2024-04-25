import os
import pandas as pd


data_dir = "data"
object_selection = []

for path in os.listdir(data_dir):
    object_selection.append(path)


df = pd.read_csv("breast-level_annotations.csv")

selected_df = df[df["study_id"].isin(object_selection)]

selected_df.to_csv("data.csv")
