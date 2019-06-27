import random
import pickle
import numpy as np

random.seed(1234)

with open('../shop_data/reviews.pkl', 'rb') as f:
  df = pickle.load(f)
  reviews_df = df[['user_id', 'item_id', 'time_stamp']]
  meta_df = df[['item_id', 'cat_id']]
  meta_df_action = df[['item_id', 'action_type']]

meta_df = meta_df.drop_duplicates(subset=['item_id'], keep='first') #保存第一条重复数据
meta_df_action = meta_df_action.drop_duplicates(subset=['item_id'], keep='first')
# print(meta_df)
# print(reviews_df)


def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))

  #为df表进行分类编号
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key
# _key:key的列表，_map:key的列表加编号


item_map, item_key = build_map(meta_df, 'item_id')
print('1:', meta_df)
cat_map, cat_key = build_map(meta_df, 'cat_id')
print('2:', meta_df)
# print('item_map:\n', item_map, '\nitem_key:\n', item_key)

item_map_action, item_key_action = build_map(meta_df_action, 'item_id')
action_map, action_key = build_map(meta_df_action, 'action_type')
# print('cat_map:\n', cat_map, '\ncat_key:\n', cat_key)

user_map, user_key = build_map(reviews_df, 'user_id')
# print('user_map:\n', user_map, '\nuser_key:\n', user_key)


user_count, item_count, cate_count, action_count, example_count =\
    len(user_map), len(item_map), len(cat_map), len(action_map), reviews_df.shape[0]
print('user_count: %d\t item_count: %d\t cat_count: %d\t action_count:%d\t example_count: %d' %
      (user_count, item_count, cate_count, action_count, example_count))

meta_df = meta_df.sort_values('item_id')
meta_df = meta_df.reset_index(drop=True)
meta_df_action = meta_df_action.sort_values('item_id')
meta_df_action = meta_df_action.reset_index(drop=True)

# 为item_id属性进行编号
reviews_df['item_id'] = reviews_df['item_id'].map(lambda x: item_map[x])
# 根据user_id、time_stamp编号进行排序（sort_values：排序函数）
reviews_df = reviews_df.sort_values(['user_id', 'time_stamp'])
# 重新建立索引
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['user_id', 'item_id', 'time_stamp']]

print(meta_df['cat_id'])
cate_list = [meta_df['cat_id'][i] for i in range(len(item_map))]
cate_list = np.array(cate_list, dtype=np.int32)
print(cate_list)

action_list = [meta_df_action['action_type'][i] for i in range(len(item_map_action))]
action_list = np.array(action_list, dtype=np.int32)

with open('../shop_data/remap.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid
  pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
  pickle.dump(action_list, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, cate_count, action_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((item_key, cat_key, action_key,  user_key), f, pickle.HIGHEST_PROTOCOL)
