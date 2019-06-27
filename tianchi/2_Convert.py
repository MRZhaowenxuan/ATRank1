import pickle
import pandas as pd
from datetime import datetime
def to_df(file_path):
  with open(file_path, 'r') as fin:
    df = {}
    i = 0
    for line in fin:
      df[i] = eval(line)
      # print("第%d条"%i)
      # print(df[i])
      i += 1
    # 把json数据转换成列表式的数据
    df = pd.DataFrame.from_dict(df, orient='index')
    # print(df)
    return df

reviews_df = to_df('../shop_data/user_log_format1.json')
reviews_df['time_stamp'] = ['2015%s' % i for i in reviews_df['time_stamp']]
# print(reviews_df)


def time2stamp(time_s):   #转时间戳函数
    time_s = datetime.strptime(time_s, '%Y%m%d')
    stamp = int(datetime.timestamp(time_s))
    return stamp

reviews_df['time_stamp'] = reviews_df['time_stamp'].apply(time2stamp)


with open('../shop_data/reviews.pkl', 'wb') as f:
  pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

# f = open('../shop_data/reviews.pkl', 'rb')
# data = pickle.load(f)
# print(data)


