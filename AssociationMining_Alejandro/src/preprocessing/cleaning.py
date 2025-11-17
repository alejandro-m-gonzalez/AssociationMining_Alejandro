from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Tuple
import pandas as pd

@dataclass
class PreprocessReport:
    total_transactions_before: int
    empty_transactions: int
    single_item_transactions: int
    duplicate_items_instances: int
    invalid_item_instances: int
    total_transactions_after: int
    total_items_after: int
    unique_products_after: int

def standardize_item(item: str) -> str:
    return ' '.join(str(item).strip().split()).lower()

def clean_transactions(transactions: List[List[str]], valid_products: Set[str]|None=None) -> Tuple[List[List[str]], PreprocessReport]:
    before = len(transactions)
    empty = 0
    single = 0
    dup_instances = 0
    invalid_instances = 0
    cleaned = []
    for t in transactions:
        items = [standardize_item(x) for x in t if str(x).strip()!='']
        if not items:
            empty += 1
            continue
        seen = set()
        no_dups = []
        for it in items:
            if it in seen:
                dup_instances += 1
            else:
                seen.add(it)
                no_dups.append(it)
        if valid_products is not None:
            filtered = [it for it in no_dups if (it in valid_products)]
            invalid_instances += len(no_dups) - len(filtered)
            no_dups = filtered
        if len(no_dups) <= 1:
            single += 1
            continue
        cleaned.append(no_dups)
    total_items_after = sum(len(t) for t in cleaned)
    unique_products_after = len(set(it for t in cleaned for it in t))
    report = PreprocessReport(
        total_transactions_before=before,
        empty_transactions=empty,
        single_item_transactions=single,
        duplicate_items_instances=dup_instances,
        invalid_item_instances=invalid_instances,
        total_transactions_after=len(cleaned),
        total_items_after=total_items_after,
        unique_products_after=unique_products_after,
    )
    return cleaned, report
