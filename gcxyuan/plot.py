from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def TSNE_plot(dataset_x,label):
    X_tsne = tsne.fit_transform(dataset_x)  # dataset [N, dim]
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne-x_min)/(x_max-x_min)  
    plt.figure(figsize=(8,8))
    colors = cm.rainbow(np.linspace(0,1,len(label)))    # label是标签，我这里每个标签的元素是一个元组
     
    #####  统计每个簇的中心坐标 ####
    group = [[] for _ in range(len(label))]
    for i in range(len(X_norm)):
        group[label[i]].append(X_norm[i])
    id_posi = []
    for i in range(len(label)):
        id_posi.append(np.mean(np.array(group[i]), 0))
    
    #id2name 是一个字典，根据id找到name
    #for i in range(len(label)):
    #    plt.text(id_posi[i][0], id_posi[i][1], id2name[label[i]], color=colors[i],fontdict={'weight': 'bold', 'size':9})
    #####----------------------#####
    for i in range(len(X_norm)):
        plt.text(X_norm[i,0], X_norm[i,1], '.', color=colors[label[i]],fontdict={'weight': 'bold', 'size':9})
    plt.xticks([])
    plt.yticks([])
    #plt.title('test')
    #plt.savefig('cluster.jpg')