import pandas as pd
import numpy as np
import glob

'''
@ Processing raw data for NN models
accs = glob.glob('row_data/*/score_per_query.tsv')
times = glob.glob('row_data/*/times.tsv')
manager = []
for i in range (0, len(accs)):
    acc = pd.read_csv(accs[i], sep='\t')
    time = pd.read_csv(times[i], sep='\t')
    acc = acc[['model', 'doc_per_query', 'ndcg@10']]
    df = pd.concat([acc, time], axis=1)
    df['nDCG'] = df['ndcg@10']
    df['time'] = df['doc_per_query'] * df['elapsed_doc']
    df['time'] = df['time'] * 10e-6
    df['qid'] = np.arange(df.shape[0])
    df = df[['qid', 'model', 'doc_per_query', 'nDCG', 'time']]
    df.to_csv(f"data/population/{accs[i].split('/')[1]}_per_query.tsv", sep='\t', index=False)
    manager.append([accs[i].split('/')[1], df['nDCG'].mean(), df['time'].sum() / df['doc_per_query'].sum()])
# pd.DataFrame(manager, columns=['model', 'nDCG', 'time']).to_csv('data/nn.tsv', sep='\t', index=False)
print(manager)
# print(df['time'].sum() / df.shape[0])
'''

files = glob.glob('data/population/*_*_per_query.tsv')
manager = []
for i in range (0, len(files)):
    df = pd.read_csv(files[i], sep='\t')
    manager.append([files[i].split('/')[2].split('_')[0] + '_' + files[i].split('/')[2].split('_')[1], df['nDCG'].mean(), df['time'].sum() / df['docs_per_query'].sum(), df['energy'].sum() / df['docs_per_query'].sum()])
pd.DataFrame(manager, columns=['model', 'nDCG', 'time', 'energy']).to_csv('data/trees_new.tsv', sep='\t', index=False)

