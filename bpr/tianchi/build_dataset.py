import random
import pickle
import numpy as np

random.seed(1234)

with open('../../shop_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  action_list = pickle.load(f)
  user_count, item_count, cate_count, action_count, example_count = pickle.load(f)

train_set = []
test_set = []
for reviewerID, hist in reviews_df.groupby('user_id'):
  pos_list = hist['item_id'].tolist()
  length = len(pos_list)
  if length < 20 | length > 150:
    continue
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg
  neg_list = [gen_neg() for i in range(len(pos_list))]
  rid_list = [reviewerID for i in range(len(pos_list))]
  hist = list(zip(rid_list, pos_list, neg_list))

  # print(len(hist))
  train_set.extend(hist[:-1])
  test_set.append(hist[-1])

random.shuffle(train_set)
random.shuffle(test_set)

# assert len(test_set) == user_count
# assert len(test_set) + len(train_set) == example_count

train_set = np.array(train_set, dtype=np.int32)
test_set = np.array(test_set, dtype=np.int32)

# print(len(train_set))
# print(len(test_set))

with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
