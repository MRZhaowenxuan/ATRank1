import numpy as np
from sli_rec_az.iterator import Iterator
import tensorflow as tf
from sli_rec_az.model import *
import random
from sli_rec_az.utils import *
from sli_rec_az.train import *

def test(train_file = "../preprocess/shop_data/train_data", test_file = "../preprocess/shop_data/test_data", save_path = "saved_model/", model_type = MODEL_TYPE, seed = SEED):
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    with tf.Session() as sess:
        train_data, test_data = Iterator(train_file), Iterator(test_file)
        user_number, item_number, cate_number = train_data.get_id_numbers()

        if model_type in MODEL_DICT: 
            cur_model = MODEL_DICT[model_type]
        else:
            print(model_type, "is not implemented")
            return
        model = cur_model(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        model_path = save_path + model_type
        model.restore(sess, model_path)
        test_auc, test_loss, test_acc = evaluate_epoch(sess, test_data, model)
        print("test_auc: %.4f, testing loss = %.4f, testing accuracy = %.4f" % (
              test_auc, test_loss, test_acc))

if __name__ == "__main__":
    test()