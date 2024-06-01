import pandas as pd
import numpy as np

df = pd.read_csv("breast-level_annotations.csv")

train_df = df[df["split"] == "training"]
test_df = df[df["split"] == "test"]

num_to_change = int(0.2 * len(train_df['split']))

# Tạo danh sách các chỉ mục ngẫu nhiên
random_indices = np.random.choice(train_df.index, num_to_change, replace=False)

# Tạo giá trị mới cho các phần tử được chọn
new_values = np.random.choice(train_df['split'].unique(), num_to_change)

# Thay đổi giá trị trong dataframe
train_df.loc[random_indices, 'split'] = "valid"

new_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

print(new_df["split"].value_counts())

new_df.to_csv("split_data.csv")