import random
import pickle

random.seed(1234)

with open('../../shop_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  action_list = pickle.load(f)
  user_count, item_count, cate_count, action_count, example_count = pickle.load(f)

data_set_pos = []
data_set_neg = []
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

  for i in range(1, len(pos_list)):
    hist = pos_list[:i]
    if i != len(pos_list) - 1:
      data_set_pos.append((reviewerID, hist, pos_list[i], 1))
      data_set_neg.append((reviewerID, hist, neg_list[i], 0))
    # else:
    #   label = (pos_list[i], neg_list[i])
    #   test_set.append((reviewerID, hist, label))

for i in range(len(data_set_pos)):
  if i % 5 == 0:
    reviewerID = data_set_pos[i][0]
    hist = data_set_pos[i][1]
    label = (data_set_pos[i][2], data_set_neg[i][2])
    test_set.append((reviewerID, hist, label))
  else:
    train_set.append(data_set_pos[i])
    train_set.append(data_set_neg[i])

random.shuffle(train_set)
random.shuffle(test_set)

#assert len(test_set) == user_count
# assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])

with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
