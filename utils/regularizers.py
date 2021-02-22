import torch 
import numpy as np
import torch.nn.functional as F
import sys

def get_per_class_weight_matrix(confusion_matrix):
    num_class, _ = confusion_matrix.shape
    per_class_weight_matrix = []
    for i in range(num_class):
        per_class_weight_list = []
        for j in range(num_class):
            per_class_weight_list.append(confusion_matrix[i,j]/sum(confusion_matrix[i,:]))
        per_class_weight_matrix.append(per_class_weight_list)
    per_class_weight_matrix = np.array(per_class_weight_matrix)
    return per_class_weight_matrix

def upper_triangle(matrix):
    upper = torch.triu(matrix, diagonal=1)
    return upper
    
def prototype_reg(args,G,F1,confusion_matrix,distance="euclid"):
    if args.net == "resnet34":
        P = F1.fc2.weight.cpu()
    elif args.net == "alexnet":
        P = F1.fc.weight.cpu()
    P = F.normalize(P,dim=1)
    per_class_weight_matrix = get_per_class_weight_matrix(confusion_matrix)
    loss_reg = 0
    n_class, n_class = confusion_matrix.shape

    if distance ==  "cosine":
        similarity = torch.mm(P, torch.transpose(P,0,1))
        similarity = torch.abs(upper_triangle(similarity))
        print(similarity)
    
    for anchor_class in range(n_class):
        class_weight = per_class_weight_matrix[anchor_class]
        for class_ in range(n_class):
            if distance == "euclid":
                loss_reg  = loss_reg + class_weight[class_]*torch.sum(torch.pow(P[anchor_class,:] - P[class_,:],2),dim=0)
            elif distance == "cosine":
                loss_reg = loss_reg + class_weight[class_]*torch.dot(P[anchor_class,:],P[class_,:])
    if distance == "euclid":
        loss_reg/= (n_class * n_class)  
        loss_reg*=10
    elif distance == "cosine":
        pass
    print(loss_reg) 
    loss_reg.backward()
    pass

def regularizer_semantic_2(args, F1,confusion_matrix,lmd=100):
    if args.net == "resnet34":
        W = F1.fc2.weight
    elif args.net == "alexnet":
        W = F1.fc.weight
    W = F.normalize(W, dim=1)
    mc = W.shape[0]
    w_expand1 = W.unsqueeze(0)
    w_expand2 = W.unsqueeze(1)
    wx = (w_expand2 - w_expand1)**2
    w_norm_mat = torch.sum((w_expand2 - w_expand1)**2, dim=-1)
    w_norm_upper = upper_triangle(w_norm_mat)
    similarity = torch.tensor(confusion_matrix).cuda()
    similarity_symm = (torch.transpose(similarity,0,1) + similarity)/2
    dist = upper_triangle(similarity_symm)
    mu = 2.0 / (mc**2 - mc) * torch.sum(w_norm_upper)
    residuals = upper_triangle((w_norm_upper - mu - dist)**2)
    rw = 2.0 / (mc**2 - mc) * torch.sum(residuals)
    return lmd * rw

def regularizer_semantic(W,embedding):
    W = F.normalize(W, dim=1)
    mc = W.shape[0]
    w_expand1 = W.unsqueeze(0)
    w_expand2 = W.unsqueeze(1)
    wx = (w_expand2 - w_expand1)**2
    w_norm_mat = torch.sum((w_expand2 - w_expand1)**2, dim=-1)
    w_norm_upper = upper_triangle(w_norm_mat)
    similarity = torch.mm(embedding, torch.transpose(embedding,0,1))
    dist = upper_triangle(similarity).clamp(min=0)
    mu = 2.0 / (mc**2 - mc) * torch.sum(w_norm_upper)
    residuals = upper_triangle((w_norm_upper - mu - dist)**2)
    rw = 2.0 / (mc**2 - mc) * torch.sum(residuals)
    return rw


if __name__ == "__main__":
    main_dict = torch.load("../save_model_ssda/MME_real_sketch.ckpt.best.pth.tar")
    W = main_dict["F1_state_dict"]["fc2.weight"].cpu()
    E = F.normalize(torch.rand([126,512]),dim=1)
    l = regularizer_semantic(W,E)
    print(l)