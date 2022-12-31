import csv

writer = open("ent_emb_final.txt", "w", encoding="utf-8")
with open(r'ent_embedding.csv', 'r') as csvfile:
    datareader = csv.reader(csvfile)
    for row in datareader:
        i=0
#        writer.write("[")
        for num in row:
            i=i+1
            if(i%16==0):
                writer.write(num + "]"+", "+ "[" )
            else:
                writer.write(num + ", " )
writer.close()