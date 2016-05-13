import io
import numpy as np
from sys import argv

def readfile(name):
    tweet_assignments = {}
    f = open(name, 'r')
    for line in f.readlines():
        parts = line.split()
        triple = (parts[3], int(parts[1]), int(parts[2]))
        if parts[0] in tweet_assignments:
            if not triple in tweet_assignments[parts[0]]:
                tweet_assignments[parts[0]].append(triple)
        else:
            tweet_assignments[parts[0]] = [triple]
    f.close()
    return tweet_assignments

def get_results(goldfile, predfile):
    gold = readfile(goldfile)
    pred = readfile(predfile)
    get_results_ass(gold, pred)

def get_best_prf(ment_ent_scrs, goldfile="../data/label-test.tsv"):
    gold = readfile(goldfile)
    ment_topent_scr, thres = {}, set([0., 1.])
    for mid in ment_ent_scrs:
        ent_lst = sorted(ment_ent_scrs[mid], key=lambda pair: pair[1], reverse=True)
        if ent_lst[0][0] == "_NULL_": continue
        ment_topent_scr[mid] = ent_lst[0]
        thres.add(ent_lst[0][1])
    thres = sorted(list(thres))
    P, R, F1 = 0., 0., 0.
    for thre in thres:
        pred = {}
        for mid in ment_topent_scr:
            pair = ment_topent_scr[mid]
            if pair[1] < thre: continue
            tid, sidx, eidx = mid
            triple = (pair[0], sidx, eidx)
            if tid in pred:
                if not triple in pred[tid]:
                    pred[tid].append(triple)
            else:
                pred[tid] = [triple]
        p, r, f = get_results_ass(gold, pred)
        if f > F1:
            P, R, F1 = p, r, f
    return P, R, F1

def get_results_ass(gold, pred):
    gold_total = 0
    predicted_total = 0
    predicted_correct = 0

    # merge keys of two dictionaries
    keys = list(set(gold.keys() + pred.keys()))
    for key in keys:
        if key not in gold:
            predicted_total = predicted_total + len(pred[key])
        elif key not in pred:
            gold_total = gold_total + len(gold[key])
        else:
            gold_ass = sorted(gold[key], key=lambda x: x[2])
            pred_ass = sorted(pred[key], key=lambda x: x[2])

            match = get_matched_count(gold_ass, pred_ass)

           # if len(pred[key]) > match:
           #     print pred[key]

            predicted_correct = predicted_correct + match
            gold_total = gold_total + len(gold[key])
            predicted_total = predicted_total + len(pred[key])

    if predicted_total == 0: return 0., 0., 0.
    precision = float(predicted_correct) / predicted_total
    recall = float(predicted_correct) / gold_total
    f1 = (2.0*precision*recall)/(precision + recall)
    return precision, recall, f1

#    print 'Correct: ' + str(predicted_correct) + ' Gold Total: ' + str(gold_total) + ' Predicted Total: ' + str(predicted_total)
#    print 'Precision: ' + str(precision) + ' Recall: ' + str(recall) + ' F1: ' + str(f1)


def get_matched_count(gold_ass, pred_ass):
    lcs_matrix = np.zeros((len(gold_ass), len(pred_ass)))

    for i in xrange(len(gold_ass)):
        for j in xrange(len(pred_ass)):
            et1 = gold_ass[i]
            et2 = pred_ass[j]

            if not (et1[1] >= et2[2] or et2[1] >= et1[2]) and (et1[0].lower() == et2[0].lower()):
                if i == 0 or j == 0:
                    lcs_matrix[i, j] = 1
                else:
                    lcs_matrix[i, j] = 1 + lcs_matrix[i - 1, j - 1]
            else:
                if i == 0 and j == 0:
                    lcs_matrix[i, j] = 0
                elif i == 0 and j != 0:
                    lcs_matrix[i, j] = max(0, lcs_matrix[i, j - 1])
                elif i != 0 and j == 0:
                    lcs_matrix[i, j] = max(lcs_matrix[i - 1, j], 0)
                elif i != 0 and j != 0:
                    lcs_matrix[i, j] = max(lcs_matrix[i - 1, j], lcs_matrix[i, j - 1])

    match_count = lcs_matrix[len(gold_ass) - 1, len(pred_ass) - 1]

    return match_count

def get_ir_prf(gold, pred):
    gc, pc, mc = 0., 0., 0.
    for g in gold:
        if gold[g][1] == "1": 
            gc += 1
            if gold[g][0] in pred[g]:
                mc += 1
                pc += 1
        else:
            if gold[g][0] in pred[g]:
                pc += 1
    if pc == 0: return 0., 0., 0.
    precision = mc / pc
    recall = mc / gc
    f1 = (2.0*precision*recall)/(precision + recall)
    return precision, recall, f1

def readirfile(fname):
    tweet_assignments = {}
    with open(fname, "rb") as f:
        for line in f:
            parts = line.split()
            tweet_assignments[parts[0]] = (parts[1], parts[2])
    return tweet_assignments

if __name__ == '__main__':
    get_results(argv[1], argv[2])
