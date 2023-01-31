import pandas as pd
import numpy as np

datasets = ['goodreads']

for dataset in datasets:
    train = pd.read_csv(f'data/{dataset}/train.tsv', sep='\t', header=None)
    train = train.rename({0:'user', 1:'item', 2:'rate'}, axis=1)
    transaction_mean = int(train.shape[0] / train['user'].nunique())
    item_count = train.groupby('item').size().reset_index(name='counts').sort_values(by='counts',
                                                                                ascending=True).reset_index(drop=True)
    item_count = item_count.rename({1: 'item', 'counts': 'pop'}, axis=1)
    most_pop = item_count.tail(transaction_mean)
    most_pop_ind = most_pop['pop'].mean() + most_pop['pop'].std()
    long_tail = item_count.head(transaction_mean)
    long_tail_ind = (long_tail['pop'].mean() + long_tail['pop'].std()) * -1
    most_pop_ind = (most_pop['pop'].mean() + most_pop['pop'].std()) * -1
    # 0: cold user/item
    # 1: warm user/item
    merged = pd.merge(train, item_count, on='item')
    merged = merged.sort_values(by=['user'])
    merged = merged.groupby('user').agg({'pop': ['mean', 'std']})
    merged = merged.xs('pop', axis=1, drop_level=True)
    merged['std'] = merged['std'].fillna(0)
    merged = merged.reset_index('user')
    merged['ind'] = (merged['mean'] + merged['std']) * -1
    # ind_v = np.append(merged['ind'].to_numpy(), [most_pop_ind, long_tail_ind])
    ind_v = (merged['ind'].to_numpy() - most_pop_ind) / (long_tail_ind - most_pop_ind)
    # f = lambda x: 0 if (x<0) else x
    # f = np.vectorize(f)
    # ind_v = f(ind_v)
    merged['APLT'] = ind_v
    merged['Recall'] = np.ones(merged.shape[0])
    merged = merged[['User', 'Recall_up', 'APLT_up']]
    merged.to_csv(f'{dataset}_utopia_point.tsv',sep='\t',index=False)


    # user_count[[0, 'group']].to_csv(f'data/{dataset}/user_groups.tsv', sep='\t', header=None, index=None)
    # item_count[[1, 'group']].to_csv(f'data/{dataset}/item_groups.tsv', sep='\t', header=None, index=None)
