from __future__ import annotations
from typing import Dict, List, Set, Tuple
from itertools import combinations

def build_vertical(transactions: List[List[str]]) -> Dict[str, Set[int]]:
    v: Dict[str, Set[int]] = {}
    for tid, t in enumerate(transactions):
        for it in t:
            v.setdefault(it, set()).add(tid)
    return v

def eclat_recursive(prefix: Tuple[str,...], items: List[Tuple[str, Set[int]]], minsup_count: int, results: Dict[frozenset, float], n_trans: int):
    for i, (item, tids) in enumerate(items):
        new_prefix = prefix + (item,)
        new_itemset = frozenset(new_prefix)
        support_count = len(tids)
        if support_count >= minsup_count:
            results[new_itemset] = support_count / n_trans
            suffix = []
            for j in range(i+1, len(items)):
                item2, tids2 = items[j]
                inter = tids & tids2
                if len(inter) >= minsup_count:
                    suffix.append((item2, inter))
            if suffix:
                eclat_recursive(new_prefix, suffix, minsup_count, results, n_trans)

def eclat(transactions: List[List[str]], min_support: float=0.2) -> Dict[frozenset, float]:
    vertical = build_vertical(transactions)
    n_trans = len(transactions)
    minsup_count = max(1, int(min_support * n_trans + 1e-9))
    items = sorted(vertical.items(), key=lambda kv: kv[0])
    results: Dict[frozenset, float] = {}
    eclat_recursive(tuple(), items, minsup_count, results, n_trans)
    return results

def generate_rules(freq: Dict[frozenset,float], min_conf: float=0.5):
    from itertools import combinations
    rules = []
    for itemset, sup in freq.items():
        if len(itemset) < 2: continue
        items = list(itemset)
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A); B = itemset - A
                if not B: continue
                conf = sup / freq.get(A, 1e-12)
                lift = conf / freq.get(B, 1e-12)
                if conf >= min_conf:
                    rules.append((A, B, sup, conf, lift))
    rules.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return rules
