import os 
csvs = os.listdir('/media/storage/liweijie/datasets/THUMOS14/thumos14_features')
train_file = open("list/train_list.txt","w")
test_file = open("list/test_list.txt","w")
for i,csv_name in enumerate(csvs):
    if i % 4 == 0:
        test_file.write(csv_name+'\n')
    else:
        train_file.write(csv_name+'\n')
