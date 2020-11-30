import numpy as np
cf_labelled = np.load("cf_labelled.npy")
cf_unlabelled = np.load("cf_unlabelled.npy")
num_class, num_class = cf_labelled.shape
labelled_acc = []
unlabelled_acc = []
for i in range(num_class):
    labelled_acc.append(cf_labelled[i,i]/sum(cf_labelled[i,:]))
    unlabelled_acc.append(cf_unlabelled[i,i]/sum(cf_unlabelled[i,:]))
f = open("acc_per_class.txt","w")

labelled_acc_np = np.array(labelled_acc)
unlabelled_acc_np = np.array(unlabelled_acc)

idxs = labelled_acc_np.argsort()
labelled_acc = labelled_acc_np[idxs]
unlabelled_acc = unlabelled_acc_np[idxs]

for x,y in zip(labelled_acc,unlabelled_acc):
    to_write  = "{:.4f} {:.4f}\n".format(float(x),float(y))
    f.write(to_write)
