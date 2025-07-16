import torch
import numpy as np
from functions import *




def SEDC (image, classifier, segments, transform=None, max_iter=20):
    # Image prediction and score
    c, p_c = pred_score(image, classifier, transform)
    # List of EdCs
    R = []
    # List of combinations to expand on
    C = []
    # List of predicted class score reduction
    P = []

    # List of segment ids
    segment_ids = np.unique(segments)

    # Support percentage of segments
    if max_iter < 1:
        iters = int(len(segment_ids) * max_iter)
        max_iter = iters

    # First loop
    for id in segment_ids:
        # Remove one segment
        new_image = remove_segment(image, segments, id)
        # New prediction and new score
        c_new, p_c_new = pred_score(new_image, classifier, transform)
        
        if c_new != c:
            # If the class changes, add the index of the region to R
            R.append(id)
        else:
            # If the class doesn't change, add the index to C and compute the score reduction
            C.append(id)
            P.append(p_c - p_c_new)
    
    # While loop if no segment alone changed the classification
    iter = 0
    while not R:
        # Picking the id of the best score reduction segment
        k = np.argmax(P)
        if iter == 0:
            best = []
            best.append(C[k])
        else:
            best = C[k]

        # All expansions of "best" with one segment
        best_set = []
        for id in segment_ids:
            if id not in best:
                temp = best.copy()
                temp.append(id)
                best_set.append(temp)
        # Pruning step
        C.pop(k)
        P.pop(k)
        # Second loop
        for id_list in best_set:
            # Remove the segments
            new_image = remove_segment(image, segments, id_list)
            # New prediction and new score
            c_new, p_c_new = pred_score(new_image, classifier, transform)
            
            if c_new != c:
                # If the class changes, add the index of the region to R
                R.append(id_list)
            else:
                # If the class doesn't change, add the index to C and compute the score reduction
                C.append(id_list)
                P.append(p_c - p_c_new)

        # Output control
        iter += 1
        print(f"Current iter {iter} (list of segments length)")
        if iter >= max_iter:
            print("Max iterations reached")
            break

    return R




def SEDCT (image, classifier, segments, target, transform=None, max_iter=20):
    # Image prediction and score
    #c, p_c = pred_score_T(image, classifier, transform)  # TODO to modify, doesn't work rn
    # Target class
    t = target
    # List of EdCs
    R = []
    # List of combinations to expand on
    C = []
    # List of difference between target class and predicted class score
    P = []

    # List of segment ids
    segment_ids = np.unique(segments)

    # Support percentage of segments
    if max_iter < 1:
        iters = int(len(segment_ids) * max_iter)
        max_iter = iters

    # First loop
    for id in segment_ids:
        # Remove one segment
        new_image = remove_segment(image, segments, id)
        # New prediction and new score
        c_new, p_c_new, p_t_new = pred_score_T(new_image, classifier, target=t, transform=transform)  
        
        if c_new == t:
            # If the class changes, add the index of the region to R
            R.append(id)
        else:
            # If the class doesn't change, add the index to C and compute the score reduction
            C.append(id)
            P.append(p_t_new - p_c_new)


    # While loop if no segment alone changed the classification
    iter = 0
    while not R:
        # Picking the id of the best score reduction segment
        k = np.argmax(P)
        if iter == 0:
            best = []
            best.append(C[k])
        else:
            best = C[k]

        # All expansions of "best" with one segment
        best_set = []
        for id in segment_ids:
            if id not in best:
                temp = best.copy()
                temp.append(id)
                best_set.append(temp)
        # Pruning step
        C.pop(k)
        P.pop(k)
        # Second loop
        for id_list in best_set:
            # Remove the segments
            new_image = remove_segment(image, segments, id_list)
            # New prediction and new score
            c_new, p_c_new, p_t_new = pred_score_T(new_image, classifier, target=t, transform=transform)
            
            if c_new != t:
                # If the class changes, add the index of the region to R
                R.append(id_list)
            else:
                # If the class doesn't change, add the index to C and compute the score reduction
                C.append(id_list)
                P.append(p_t_new - p_c_new)

        # Output control
        iter += 1
        print(f"Current iter {iter} (list of segments length)")
        if iter >= max_iter:
            print("Max iterations reached")
            break

    return R
    





# Implementazione in 
# Eventualmente includere l'algoritmo esatto per identificare il minor numero di segmenti -> poi comparison con greedy