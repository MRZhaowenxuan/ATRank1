import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../shop_data/remap.pkl', 'rb') as f:
  reviews_df = pickle.load(f)
  cate_list = pickle.load(f)
  action_list = pickle.load(f)
  user_count, item_count, cate_count, action_count, example_count = pickle.load(f)

print('user_count:%d,item_count:%d,example_count:%d' % (user_count, item_count,  example_count))

user_num = np.array(np.arange(user_count))
hist_length = []
for reviewerID, hist in reviews_df.groupby('user_id'):
  hist_list = hist['item_id'].tolist()
  print('%d:%d' % (reviewerID, len(hist_list)))
  hist_length.append(len(hist_list))

fig = plt.figure()
fig, ax = plt.subplots()
ax = fig.add_subplot(3, 2, 1)
ax.scatter(user_num, hist_length)
ax.set_xlabel('user_id')
ax.set_ylabel('hist_length')
plt.show()


action0 = 0
action1 = 0
action2 = 0
action3 = 0
for action in reviews_df['item_id']:
    if action_list[action] == 0:
        action0 += 1
    if action_list[action] == 1:
        action1 += 1
    if action_list[action] == 2:
        action2 += 1
    if action_list[action] == 3:
        action3 += 1

print('click:%d,add-to-cart:%d,purchase:%d,add-to-favourite:%d' % (action0, action1, action2, action3))

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=80)

# 再创建一个规格为 1 x 1 的子图
plt.subplot(1, 1, 1)

# 柱子总数
N = 4
# 包含每个柱子对应值的序列
values = (action0, action1, action2, action3)

# 包含每个柱子下标的序列
index = np.arange(N)

# 柱子的宽度
width = 0.35

# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index, values, width, label="rainfall", color="#87CEFA")

# 设置横轴标签
plt.xlabel('action_type')
# 设置纵轴标签
plt.ylabel('action_type_num')

# 添加标题
plt.title('distribution of actions')

# 添加纵横轴的刻度
plt.xticks(index, ('click', 'add-to-cart', 'purchase', 'add-to-favourite'))
plt.yticks(np.arange(0, 81, 10))

# 添加图例
plt.legend(loc="upper right")

plt.show()