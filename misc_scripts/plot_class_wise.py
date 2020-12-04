import numpy as np

def plt_class_wise(cf_labelled,cf_unlabelled, pred_matrix,confidence_matrix):
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

    for i in range(6):
        pred_matrix[i] = pred_matrix[i][idxs]
    i = 0
    for i in range(4):
        confidence_matrix[i] = confidence_matrix[i][idxs]
    i = 0

    for x,y in zip(labelled_acc,unlabelled_acc):
        to_write  = "{:.4f} {:.4f} {:3d} {:3d} {:3d} {:3d} {:3d} {:3d}\n".format(float(x),float(y),int(pred_matrix[0][i]), int(pred_matrix[1][i]), int(pred_matrix[2][i]), int(pred_matrix[3][i]), int(pred_matrix[4][i]),int(pred_matrix[5][i]))
        i = i + 1
        f.write(to_write)


def main():
    cf_labelled = np.load("cf_labelled_target.npy")
    cf_unlabelled = np.load("cf_unlabelled_target.npy")
    pred_matrix = np.load("pred_matrix.npy")
    confidence_matrix = np.load("confidence_matrix.npy")
    plt_class_wise(cf_labelled,cf_unlabelled, pred_matrix,confidence_matrix)

if __name__ == "__main__":
    main()


