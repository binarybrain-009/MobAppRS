"""
extract postive and negative samples from kg
"""
import numpy as np
data_path = "../datas/"

def extract_pos_neg_sample():
    AppH_positive_AppT_dict = dict()  # positive AppH->AppT sets
    apps=set()
    relations=set()
    for i in range(0,1793):
        apps.add(i)
    for i in range(0,12):
        relations.add(i)
    writer = open(data_path + "relation_exp1-test.txt", "w", encoding="utf-8")
    with open(data_path + "relation_exp1.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = line.replace("\n", "").split("\t")
            print(data)
            if data[0] not in AppH_positive_AppT_dict:
                AppH_positive_AppT_dict[data[0]] = set()
            AppH_positive_AppT_dict[data[0]].add(data[1])
            writer.write(str(data[0]) + "\t" + str(data[1]+"\t" + "1" + "\n"))  # writing positive samples
    
    for AppH_index, pos_app_set in AppH_positive_AppT_dict.items():
        unwatched_set = relations - pos_app_set
        for app_index in np.random.choice(list(unwatched_set), size=len(pos_app_set), replace=False):  # writing negative samples
            writer.write(str(AppH_index) + "\t" + str(app_index) + "\t" + "0" + "\n")
    writer.close()
    print('number of AppHeads: %d' % len(AppH_positive_AppT_dict))
    print('number of AppTails: %d' % len(apps))
    return AppH_positive_AppT_dict

if __name__ == '__main__':
    extract_pos_neg_sample()