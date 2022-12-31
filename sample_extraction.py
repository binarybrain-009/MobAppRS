"""
extract postive and negative samples from kg
"""
import numpy as np
data_path = "../datas/"

def read_data():
    entity2id = dict() # entity->id，entity contains app
    apps = set()  # app sets
    with open(data_path + "entity2id.txt", "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            data = line.replace("\n", "").split("\t")
            entity2id[data[0]] = int(data[1])
    with open(data_path + "app_id.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = line.replace("\n", "").split("\t")
            apps.add(int(data[0]))
    return entity2id, apps

def extract_pos_neg_sample():
    user_positive_app_dict = dict()  # positive user->app sets
    writer = open(data_path + "user_app.txt", "w", encoding="utf-8")
    entity2id, apps = read_data()
    with open(data_path + "user_app_kg.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = line.replace("\n", "").split("\t")
            if entity2id[data[0]] not in user_positive_app_dict:
                user_positive_app_dict[entity2id[data[0]]] = set()
            user_positive_app_dict[entity2id[data[0]]].add(entity2id[data[2]])    
            writer.write(str(entity2id[data[0]]) + "\t" + str(entity2id[data[2]]) + "\t" + "1" + "\n")  # writing positive samples
    for user_index, pos_app_set in user_positive_app_dict.items():
        unwatched_set = apps - pos_app_set
        for app_index in np.random.choice(list(unwatched_set), size=len(pos_app_set), replace=False):  # writing negative samples
            writer.write(str(user_index) + "\t" + str(app_index) + "\t" + "0" + "\n")
    writer.close()
    print('number of users: %d' % len(user_positive_app_dict))
    print('number of apps: %d' % len(apps))
    return user_positive_app_dict

if __name__ == '__main__':
    read_data()
    extract_pos_neg_sample()