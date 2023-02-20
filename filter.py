import csv as csv
import pandas as pd

dataset = "aggression_parsed_dataset.csv"
rws = pd.read_csv("./dataset/"+dataset, sep=",")
#print(rws)
thr = 1 # < 100% per label 0
label = 0 # label of row
del rws["index"]
todelete = []

newdf = pd.DataFrame(columns=["Text","ed_label_0","ed_label_1","oh_label"])


for index, row in rws.iterrows():
    if row["ed_label_0"] < thr:
            if row["oh_label"] == label:
                #print(row)
                todelete.append(index)

#print(str(todelete))
#print(rws)
for i in todelete:
   rws = rws.drop(i)

print(str(len(rws.index)))
conteggio0 = 0
totale = len(rws.index)

for index, row in rws.iterrows():
    if row["oh_label"] == 0:
        conteggio0 += 1


percentuale = (conteggio0 * 100)/totale


percentuale = round(percentuale, 2)

percentuale1  = 100 - percentuale

print("percentuale di classe 0: " + str(percentuale))

print("percentuale di classe 1: " + str(percentuale1))


rws.to_csv("./"+dataset+"_clean.csv", sep=";")
print(rws)
