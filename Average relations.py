import numpy as np
data_path = "../datas/"

app_number=1790
#writer = open(data_path + "kg_final.txt", "w", encoding="utf-8")
# with open(data_path + "kg_final.txt", "r", encoding="utf-8") as f:
relation=[]
relation_freq = {}
with open(data_path + "kg_final.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        data = line.replace("\n", "").split(" ")
        key = f'{data[0]}-{data[1]}'
        if relation_freq.get(key): 
            relation_freq[key] += 1
        else:
            relation_freq[key] = 1
sum=0
for pair, a in relation_freq.items():
    print(pair, a)
    sum = sum+a
avg=(sum*2)/(app_number*(app_number-1))
print(avg)
