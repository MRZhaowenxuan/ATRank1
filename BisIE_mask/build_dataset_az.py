import random
import pickle
import numpy as np

random.seed(1234)

with open('../raw_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count, example_count = pickle.load(f)

# [1, 2) = 0, [2, 4) = 1, [4, 8) = 2, [8, 16) = 3...  need len(gap) hot
gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
#gap = [2, 7, 15, 30, 60,]
#gap.extend( range(90, 4000, 200) )
#gap = np.array(gap)
#print(gap.shape[0])

def proc_time_emb(hist_t, cur_t):
  hist_t = [cur_t - i + 1 for i in hist_t]
  hist_t = [np.sum(i >= gap) for i in hist_t]
  return hist_t

def proc_time(cur_t,fut_t):
  fut_t = [i - cur_t + 1 for i in fut_t]
  fut_t = [np.sum(i >= gap) for i in fut_t]
  return fut_t

data_set_pos = []
data_set_neg = []
train_set = []
test_set = []
# print(reviews_df)
for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()#asin属性的值
  #时间数据的处理
  tim_list = hist['unixReviewTime'].tolist()
  tim_list = [i // 3600 // 24 for i in tim_list]
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, item_count-1)
    return neg


  neg_list = [gen_neg() for i in range(len(pos_list))]

  for i in range(1, len(pos_list)):
    hist_i = pos_list[:i]
    hist_t = proc_time_emb(tim_list[:i], tim_list[i])
    fut_t = proc_time(tim_list[i], tim_list[i+1:len(tim_list)])

    if i != len(pos_list) - 1:
      fut_list = pos_list[i+1:len(pos_list)]
      action_i = hist_i + fut_list
      action_t = hist_t + fut_t

      # data_set_pos.append((reviewerID, hist_i, hist_t, pos_list[i], 1, fut_list, fut_t))
      # data_set_neg.append((reviewerID, hist_i, hist_t, neg_list[i], 0, fut_list, fut_t))

      data_set_pos.append((reviewerID, action_i, action_t, pos_list[i], 1))
      data_set_neg.append((reviewerID, action_i, action_t, neg_list[i], 0))


for i in range(len(data_set_pos)):
  if i % 5 == 0:
    reviewerID = data_set_pos[i][0]
    action_i = data_set_pos[i][1]
    action_t = data_set_pos[i][2]
    label = (data_set_pos[i][3], data_set_neg[i][3])
    test_set.append((reviewerID, action_i, action_t, label))
  else:
    train_set.append(data_set_pos[i])
    train_set.append(data_set_neg[i])

random.shuffle(train_set)
random.shuffle(test_set)

#print(test_set)
#assert len(test_set) == user_count
#assert(len(test_set) + len(train_set) // 2 == reviews_df.shape[0])
print(len(train_set))
print(len(test_set))
# print(train_set[100])
# print(test_set[50])
with open('dataset.pkl', 'wb') as f:
  pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count), f, pickle.HIGHEST_PROTOCOL)
