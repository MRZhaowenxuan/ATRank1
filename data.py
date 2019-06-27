import pickle

with open('../shop_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  action_list = pickle.load(f)
  user_count, item_count, cate_count, action_count, example_count = pickle.load(f)

print('user_count:%d,item_count:%d,example_count:%d' % (user_count, item_count,  example_count))
for reviewerID, hist in reviews_df.groupby('user_id'):
  hist_list = hist['item_id'].tolist()
  print('%d:%d' % (reviewerID, len(hist_list)))

action0 = 0
action1 = 0
action2 = 0
action3 = 0
for action in action_list:
    if action == 0:
        action0 += 1
    if action == 1:
        action1 += 1
    if action == 2:
        action2 += 1
    if action == 3:
        action3 += 1

print('action0:%d,action1:%d,action2:%d,action3:%d' % (action0, action1, action2, action3))