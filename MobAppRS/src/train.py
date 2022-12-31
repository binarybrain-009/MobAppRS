#import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
from model import KGEP
tf.disable_v2_behavior()

np.random.seed(555)
tf.set_random_seed(1)
def train(args, data, show_loss, show_topk):
    n_AppHead, n_AppTail, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    model = KGEP(args, n_AppHead, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    AppHead_list, train_record, test_record, AppTail_set, k_list = topk_settings(show_topk, train_data, test_data, n_AppTail) #it will select 
    #topk from the entire list k=[1,3,5,7] or [10,20,30,40] the v

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            # train_precision, train_recall = ctr_eval(sess, model, train_data, args.batch_size)
            # eval_precision, eval_recall = ctr_eval(sess, model, eval_data, args.batch_size)
            # test_precision, test_recall = ctr_eval(sess, model, test_data, args.batch_size)
            print('epoch %d' % step)
            # print('epoch %d    train precision: %.4f  recall: %.4f   train map: %.4f     eval precision: %.4f  recall: %.4f eval map: %.4f    test precision: %.4f  recall: %.4f  test map: %.4f '
            #       % (step, train_precision, train_recall, eval_precision, eval_recall, test_precision, test_recall))

            # top-K evaluation
            if show_topk:
                precision, recall, mapN = topk_eval(
                    sess, model, AppHead_list, train_record, test_record, AppTail_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.8f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.8f\t' % i, end='')
                print()
                print('mapN: ', end='')
                for i in mapN:
                    print('%.8f\t' % i, end='')
                print('\n')


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        AppHead_num = 100
#        k_list = [10, 20, 30, 40]
        k_list = [1,3,5,7]
        #rename to get_App_Record
        train_record = get_AppHead_record(train_data, True)#get app record  # all positive and negative samples
        test_record = get_AppHead_record(test_data, False) #get app record  # test set 
        #all samples of training set and only positive samples of test set are taken FROM get_App_Record
        #rename to AppH_list
        AppHead_list = list(set(train_record.keys()) & set(test_record.keys()))  # common apps of training set and test set
        if len(AppHead_list) > AppHead_num:
            AppHead_list = np.random.choice(AppHead_list, size=AppHead_num, replace=False)
        #Rename to AppT_list
        AppTail_set = set(list(range(n_item))) #tail_set or app2_Set or appT_set
        return AppHead_list, train_record, test_record, AppTail_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.AppH_indices: data[start:end, 0],
                 model.AppT_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    precision_list = []
    recall_list = []
    map_list = []
    while start + batch_size <= data.shape[0]:
        precision, recall, map = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        precision_list.append(precision)
        recall_list.append(recall)
        map_list.append(map)
        start += batch_size
    return float(np.mean(precision_list)), float(np.mean(recall_list)), float(np.mean(map_list))


def precision(rank, ground_truth):
    hits = [1 if item in ground_truth else 0 for item in rank]
    result = np.cumsum(hits, dtype=np.float32)/np.arange(1, len(rank)+1)
    return result


def maps(rank, ground_truth):
    pre = precision(rank, ground_truth)
    pre = [pre[idx] if item in ground_truth else 0 for idx, item in enumerate(rank)]
    sum_pre = np.cumsum(pre, dtype=np.float32)
    relevant_num = np.cumsum([1 if item in ground_truth else 0 for item in rank])
    result = [p / r_num if r_num != 0 else 0 for p, r_num in zip(sum_pre, relevant_num)]
#    print(1)
    return result


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    map_list = {k: [] for k in k_list}
    for user in user_list:
        test_item_list = list(item_set - train_record[user])#excluding all the apps linked to the user in the training set to create  a test set
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.AppH_indices: [user] * batch_size,#user-Daksh [Daksh*5], 
                                                    model.AppT_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score#create scoere map
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.AppH_indices: [user] * batch_size,
                       model.AppT_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]#sort by score then take its key  i.e. the item

        for k in k_list:#[10, 20, 20, 40]
            hit_num = len(set(item_sorted[:k]) & test_record[user])  # hit nums
            precision_list[k].append(hit_num / k)#(TP/TP+FP) DENOMINATOR ACTUAL RESULTS
            recall_list[k].append(hit_num / len(test_record[user]))#(TP/TP+FN) Denominator predicted results
            # print(map_list[k])
            # print(" ")
            # print(  [user])
#            print(k-1)
#            print(maps(item_sorted, test_record[user])[k-1])
            map_list[k].append(maps(item_sorted, test_record[user])[k-1])


    precisionss = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    map_1 = [np.mean(map_list[k]) for k in k_list]
    return precisionss, recall, map_1


def get_AppHead_record(data, is_train):  # all samples of training set and positive samples of test set are reserved
    AppHead_history_dict = dict()
    for interaction in data:
        AppHead = interaction[0]
        AppTail = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if AppHead not in AppHead_history_dict:
                AppHead_history_dict[AppHead] = set()
            AppHead_history_dict[AppHead].add(AppTail)
    return AppHead_history_dict
