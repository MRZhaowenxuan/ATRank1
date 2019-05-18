import numpy as np

class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i+1) * self.batch_size, len(self.data))]

        self.i += 1

        u, i, y, sl_h, sl_f = [], [], [], [], []
        for t in ts:
            u.append(t[0])#user id
            i.append(t[3])#post_list[i]未来的行为
            y.append(t[4])#0 or 1
            sl_h.append(len(t[1]))#hist_i的长度
            sl_f.append(len(t[5]))
        max_sl_h = max(sl_h)
        max_sl_f = max(sl_f)

        hist_i = np.zeros([len(ts), max_sl_h], np.int64)
        hist_t = np.zeros([len(ts), max_sl_h], np.float32)
        fut_i = np.zeros([len(ts), max_sl_f], np.int64)
        fut_t = np.zeros([len(ts), max_sl_f], np.float32)

        k = 0
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                hist_t[k][l] = t[2][l]
            k += 1
        k = 0
        for t in ts:
            for l in range(len(t[5])):
                fut_i[k][l] = t[5][l]
                fut_t[k][l] = t[6][l]
            k += 1

        return self.i, (u, i, y, hist_t, hist_i, sl_h, fut_i, fut_t, sl_f)


class DataInputTest:

    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i+1) * self.batch_size, len(self.data))]

        self.i += 1

        u, i, j, sl_h, sl_f = [], [], [], [], []

        for t in ts:
            u.append(t[0])
            i.append(t[3][0])
            j.append(t[3][1])
            sl_h.append(len(t[1]))
            sl_f.append(len(t[4]))

        max_sl_h = max(sl_h)
        max_sl_f = max(sl_f)
        #print(max_sl)

        hist_i = np.zeros([len(ts), max_sl_h], np.int64)
        hist_t = np.zeros([len(ts), max_sl_h], np.float32)
        fut_i = np.zeros([len(ts), max_sl_f], np.int64)
        fut_t = np.zeros([len(ts), max_sl_f], np.float32)

        k = 0
        #print(ts)
        for t in ts:
            for l in range(len(t[1])):
                hist_i[k][l] = t[1][l]
                hist_t[k][l] = t[2][l]
            k += 1

        k = 0
        for t in ts:
            for l in range(len(t[4])):
                fut_i[k][l] = t[4][l]
                fut_t[k][l] = t[5][l]
            k += 1

        return self.i, (u, i, j, hist_t, hist_i, sl_h, fut_i, fut_t, sl_f)