from __future__ import annotations
import pandas as pd
from typing import List

def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get('id', list(df.columns)[0])
    name_col = cols.get('name', list(df.columns)[1])
    df = df.rename(columns={id_col:'id', name_col:'name'})
    return df[['id','name']].astype({'id':str, 'name':str})

def load_transactions(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def to_transaction_list(df: pd.DataFrame) -> list[list[str]]:
    cols = [c.lower() for c in df.columns]
    lower_map = dict(zip(df.columns, cols))
    if 'items' in cols:
        items_col = [k for k,v in lower_map.items() if v=='items'][0]
        return [[x.strip() for x in str(v).split(',') if str(x).strip()] for v in df[items_col].fillna('')]
    else:
        item_cols = df.columns
        txns = []
        for _, row in df.iterrows():
            items = []
            for c in item_cols:
                val = row[c]
                if isinstance(val, (int, float)) and val != 0:
                    items.append(str(c))
                elif isinstance(val, str) and val.strip():
                    items.append(val.strip())
            txns.append(items)
        return txns
