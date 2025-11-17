import streamlit as st
import pandas as pd
import numpy as np
import time, psutil
import matplotlib.pyplot as plt
from utils.io import load_products, load_transactions, to_transaction_list
from preprocessing.cleaning import clean_transactions, PreprocessReport, standardize_item
from algorithms.apriori import apriori as apriori_fit, generate_rules as apriori_rules
from algorithms.eclat import eclat as eclat_fit, generate_rules as eclat_rules

st.set_page_config(page_title='Supermarket Association Mining', layout='wide')
st.title('🛒 Interactive Supermarket + Association Rule Mining')
st.caption('Apriori & Eclat implemented from scratch. Clean → Mine → Recommend.')

with st.sidebar:
    st.header('Dataset')
    default_products_path = 'data/products.csv'
    default_txn_path = 'data/sample_transactions.csv'
    prod_file = st.file_uploader('Upload products.csv', type=['csv'], key='prod')
    if prod_file is None:
        st.info('Using default products file', icon='🗂️')
        products_df = load_products(default_products_path)
    else:
        products_df = load_products(prod_file)
    st.write(f'**Loaded products:** {len(products_df)}')
    valid_names = set(products_df['name'].map(lambda x: ' '.join(str(x).strip().split()).lower()).tolist())
    valid_ids = set(products_df['id'].astype(str).tolist())
    valid_all = valid_names.union(valid_ids)
    txn_file = st.file_uploader('Upload transactions CSV', type=['csv'], key='txn')
    load_default = st.checkbox('Use provided sample_transactions.csv', value=True)
    if txn_file is None and load_default:
        transactions_df = load_transactions(default_txn_path)
    elif txn_file is not None:
        transactions_df = load_transactions(txn_file)
    else:
        transactions_df = pd.DataFrame({'items': []})

raw_txns = to_transaction_list(transactions_df)

st.subheader('1) Build Transactions Manually')
with st.expander('Open manual transaction builder', expanded=False):
    items = products_df['name'].tolist()[:30]
    cols = st.columns(5)
    if 'manual_current' not in st.session_state:
        st.session_state.manual_current = []
    if 'manual_all' not in st.session_state:
        st.session_state.manual_all = []
    for idx, it in enumerate(items):
        if cols[idx % 5].button(it):
            st.session_state.manual_current.append(it)
    st.write('Current selection:', st.session_state.manual_current)
    add_btn = st.button('➕ Add this as a transaction')
    clear_btn = st.button('🧹 Clear selection')
    if add_btn:
        st.session_state.manual_all.append(st.session_state.manual_current.copy())
        st.session_state.manual_current = []
        st.success('Added transaction.')
    if clear_btn:
        st.session_state.manual_current = []
    if st.session_state.manual_all:
        st.write('Manual transactions so far:')
        st.dataframe(pd.DataFrame({'items': [', '.join(t) for t in st.session_state.manual_all]}))

all_txns = raw_txns + st.session_state.get('manual_all', [])

st.subheader('2) Preprocess Data')
do_validate = st.checkbox('Validate items against products list', value=True)
cleaned_txns, report = clean_transactions(all_txns, valid_all if do_validate else None)
c1, c2 = st.columns(2)
with c1:
    st.markdown('**Before Cleaning (sample):**')
    st.dataframe(pd.DataFrame({'items': [', '.join(t) for t in all_txns[:20]]}))
with c2:
    st.markdown('**After Cleaning (sample):**')
    st.dataframe(pd.DataFrame({'items': [', '.join(t) for t in cleaned_txns[:20]]}))
rep_df = pd.DataFrame({
    'Metric': [
        'Total transactions (before)',
        'Empty transactions (removed)',
        'Single-item transactions (removed)',
        'Duplicate items (instances)',
        'Invalid items (instances)',
        'Valid transactions (after)',
        'Total items (after)',
        'Unique products (after)',
    ],
    'Value': [
        report.total_transactions_before,
        report.empty_transactions,
        report.single_item_transactions,
        report.duplicate_items_instances,
        report.invalid_item_instances,
        report.total_transactions_after,
        report.total_items_after,
        report.unique_products_after,
    ]
})
st.table(rep_df)

st.subheader('3) Mine Association Rules')
ms = st.slider('Minimum support', 0.01, 0.9, 0.2, 0.01)
mc = st.slider('Minimum confidence', 0.05, 0.95, 0.5, 0.05)
if not cleaned_txns:
    st.warning('No valid transactions after preprocessing. Please add/import data.')
    st.stop()
import time
perf_rows = []
start = time.time()
ap_freq = apriori_fit(cleaned_txns, min_support=ms)
ap_rules = apriori_rules(ap_freq, min_conf=mc)
ap_time = (time.time() - start) * 1000.0
perf_rows.append(['Apriori', f'{ap_time:.1f}', len(ap_rules)])
start = time.time()
ec_freq = eclat_fit(cleaned_txns, min_support=ms)
ec_rules = eclat_rules(ec_freq, min_conf=mc)
ec_time = (time.time() - start) * 1000.0
perf_rows.append(['Eclat', f'{ec_time:.1f}', len(ec_rules)])
perf_df = pd.DataFrame(perf_rows, columns=['Algorithm', 'Exec time (ms)', '# rules'])
st.markdown('**Performance comparison**')
st.dataframe(perf_df, hide_index=True, use_container_width=True)

st.subheader('4) Recommendations (user-friendly)')
def rules_to_map(rules):
    m = {}
    for A,B,sup,conf,lift in rules:
        if len(A)!=1:
            continue
        a = next(iter(A))
        target = next(iter(B)) if len(B)==1 else ', '.join(sorted(B))
        m.setdefault(a, []).append((target, conf, sup, lift))
    for k in m:
        m[k].sort(key=lambda x: x[1], reverse=True)
    return m
ap_map = rules_to_map(ap_rules)
ec_map = rules_to_map(ec_rules)
all_keys = sorted(set(ap_map.keys()) | set(ec_map.keys()))
selected_algo = st.radio('Which algorithm to use for recommendations?', ['Apriori','Eclat'], horizontal=True)
choice = st.selectbox('Pick a product:', options=all_keys)
if choice:
    src_map = ap_map if selected_algo=='Apriori' else ec_map
    recs = src_map.get(choice, [])
    if not recs:
        st.info('No associated products found at current thresholds.')
    else:
        df = pd.DataFrame([
            {
                'Also bought': r[0],
                'Strength (%)': int(round(r[1]*100)),
                'Support': round(r[2],3),
                'Confidence': round(r[1],3),
                'Lift': round(r[3],3),
            } for r in recs[:10]
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)
        fig, ax = plt.subplots()
        ax.bar(df['Also bought'], df['Strength (%)'])
        ax.set_ylabel('Strength (%)')
        ax.set_xticklabels(df['Also bought'], rotation=30, ha='right')
        st.pyplot(fig)
        st.caption('Tip: Co-locate top pairs and test bundles.')
with st.expander('Technical details: frequent itemsets & rules'):
    st.write('**Apriori** frequent itemsets:', len(ap_freq))
    st.write('**Eclat** frequent itemsets:', len(ec_freq))
    st.write('Example Apriori rules (top 10):')
    st.dataframe(pd.DataFrame([
        {
            'A': ', '.join(sorted(list(a))),
            'B': ', '.join(sorted(list(b))),
            'support': round(s,3),
            'confidence': round(c,3),
            'lift': round(l,3),
        } for a,b,s,c,l in ap_rules[:10]
    ]))
st.success('✅ Ready. Use the sliders and selector to explore rules.')
