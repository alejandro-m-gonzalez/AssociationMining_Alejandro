from __future__ import annotations
from typing import List, Dict, Tuple, Set
from itertools import combinations

def calc_support(itemset: frozenset, transactions: List[Set[str]]) -> float:
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def apriori(transactions: List[List[str]], min_support: float=0.2) -> Dict[frozenset, float]:
    trans_sets = [set(t) for t in transactions]
    n = len(trans_sets)
    item_counts: Dict[str,int] = {}
    for t in trans_sets:
        for it in t:
            item_counts[it] = item_counts.get(it,0)+1
    Lk = {frozenset([i]): c/n for i,c in item_counts.items() if c/n >= min_support}
    freq: Dict[frozenset,float] = dict(Lk)
    k = 2
    while Lk:
        prev = list(Lk.keys())
        candidates = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                u = prev[i].union(prev[j])
                if len(u)==k:
                    if all(frozenset(s) in Lk for s in combinations(u, k-1)):
                        candidates.add(frozenset(u))
        sup_k: Dict[frozenset,float] = {}
        for cand in candidates:
            sup = calc_support(cand, trans_sets)
            if sup >= min_support:
                sup_k[cand] = sup
        Lk = sup_k
        freq.update(Lk)
        k += 1
    return freq

def generate_rules(freq: Dict[frozenset,float], min_conf: float=0.5):
    rules = []
    for itemset, sup in freq.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A)
                B = itemset - A
                if not B:
                    continue
                conf = sup / freq.get(A, 1e-12)
                lift = conf / freq.get(B, 1e-12)
                if conf >= min_conf:
                    rules.append((A, B, sup, conf, lift))
    rules.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return rules
