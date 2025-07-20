# Currently we have a horrendous split for training and testing data. 80/20 > 3001/1 
import splitfolders
dr = 'dataset/asl_alphabet_train/asl_alphabet_train'
splitfolders.ratio(dr,"splitdataset" ,ratio=(0.8,0.2))
