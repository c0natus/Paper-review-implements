import numpy as np

def get_Recall(rank_list, target_item_list):        
        hit_list = np.intersect1d(target_item_list, rank_list)        
        return len(hit_list) / len(target_item_list)

    
def _get_DCG(rank_list, target_item_list):
    DCG = []
    for i in range(len(rank_list)):
        item = rank_list[i]
        if item in target_item_list:
            DCG.append(1/np.log(i+2))
    
    return sum(DCG)
    

def _get_IDCG(target_item_list):
    IDCG = []
    for i in range(len(target_item_list)):
        IDCG.append(1/np.log(i+2))
    
    return sum(IDCG)
        

def get_NDCG(rank_list, target_item_list):
    DCG = _get_DCG(rank_list, target_item_list)
    IDCG = _get_IDCG(target_item_list)      
      
    return DCG/IDCG