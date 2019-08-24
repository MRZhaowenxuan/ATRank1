import random
import pickle
import numpy as np

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

data = []
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  if len(pos_list) >= 3:
    data.append(pos_list)

# print(data)
random.shuffle(data)


with open('dataset.pkl', 'wb') as f:
  pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count), f, pickle.HIGHEST_PROTOCOL)
