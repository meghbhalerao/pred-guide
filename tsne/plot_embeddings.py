import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

n_classes  = 10
tsne_embeddings = np.load("tsne_embeddings.npy")
labels = np.load('labels.npy')
labels = list(labels)
"""
for i in range(n_classes):
    labels.append(-50)
weights = np.load('weights.npy')
n_examples, _ =  tsne_embeddings.shape
x = list(tsne_embeddings[0:n_examples - n_classes,0])
y = list(tsne_embeddings[0:n_examples - n_classes,1])
labels_xy = list(weights[0:n_examples - n_classes])
"""

x = list(tsne_embeddings[:,0])
y = list(tsne_embeddings[:,1])
labels_xy = labels

"""
print(labels)
class_labels = [4,5,9,12,78,83,92,97,105,124]
prototype_x = list(tsne_embeddings[n_examples - n_classes:n_examples,0])
prototype_y = list(tsne_embeddings[n_examples - n_classes:n_examples,1])
labels_prototype = list(weights[n_examples - n_classes: n_examples])
values = [0,1,-50]
"""

print(len(x))
print(len(y))
print(len(labels_xy))

"""
print(len(labels_xy))
print(len(prototype_x))
print(len(prototype_y))
print(len(labels_prototype))
print(labels_prototype)
"""
colors = ['blue','red','green','yellow','purple','lawngreen','violet','indigo','lime','orange']

plt.scatter(x, y, c=labels_xy, cmap=mpl.colors.ListedColormap(colors), marker='+')

"""
for idx, label in enumerate(labels):
    for i in range(len(class_labels)):
        if label == class_labels[i]:
            plt.scatter(x[idx],y[idx],marker = '+',color=colors[i])



plt.scatter(prototype_x,prototype_y,marker = 'o', color='black')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) #
plt.tick_params(
axis='y',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=False,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=False) 
"""
#plt.scatter(x, y, c=labels, cmap='viridis', marker = 'x')
#plt.title('ours')
#plt.title("t-SNE of MME for 4096 dimensional features", fontsize = 7)
#plt.colorbar()
#plt.box(False)
plt.show()
#weights = np.round(weights,decimals=2)
#weights = (np.exp(weights) - 1)/(np.exp(1) - 1)
#weights = list(weights)
#print(weights)
#print(min(weights),max(weights))
#norm = mpl.colors.Normalize(vmin=weights.min(), vmax=weights.max())
#cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
#cmap.set_array(weights)

#for i, shade in enumerate(weights):
#    plt.scatter(x[i],y[i],color=cmap.to_rgba(i + 1),marker = '+')


"""
minority = ['ceiling_fan', 'cell_phone', 'pencil', 'carrot', 'aircraft_carrier', 'castle', 'power_outlet', 'calculator', 'cello', 'toe']
minority_idx = [0,20,26,27,29,30,31,83,89,118]
majority = ['whale', 'snake', 'bird', 'skateboard', 'tiger', 'swan', 'speedboat', 'shoe', 'spider', 'see_saw']
majority_idx = [11,98,100,101,102,103,104,110,117,124]
#majority_idx = [0,1,2,3,4,5,6,7,8,9]
minority_idx = [20,24,29,32,35,36,55,61,98,118]
colors_list = ['green','blue','red','orange','lawngreen']
color = []
color_flag = 0
prev_label = minority_idx[0]
for idx, label in enumerate(labels):
    if prev_label != label:
        prev_label = label
colors = ['green','blue','red','orange','purple']

# Dropping the non-majority and non-minority classes
for idx, label in enumerate(labels):
    for i in range(len(minority_idx)):
        if label == minority_idx[i]:
            plt.scatter(x[idx],y[idx],marker = '+',color=colors[i])
"""
#plt.scatter(x, y, c=weights, cmap='Reds', marker='+')
#plt.scatter(prototype_x,prototype_y, color = 'black', marker = '+')


#colors = ['green','blue','red','orange','purple','yellow','cyan','pink','gold','lawngreen']
#plt.scatter(x, y, c=labels, cmap=mpl.colors.ListedColormap(colors), marker='+')