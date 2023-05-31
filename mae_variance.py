import pandas as pd
import glob

if __name__ == '__main__':
    '''
    files = glob.glob(f'data/population/msn/*_*_per_query*')
    manager = []
    for file in files:
        df = pd.read_csv(file, sep='\t')
        df['nDCG_d'] = (1 - df['nDCG']) ** 2
        df['time_d'] = (0 - df['time']) ** 2
        df['energy_d'] = (0 - df['energy']) ** 2
        df['sum_d'] = df['nDCG_d'] + df['time_d'] + df['energy_d']
        df['sum_d'] = df['sum_d'] ** 0.5
        manager.append([file, df['sum_d'].mean()])
    res = pd.DataFrame(manager, columns=['model', 'ed'])
    res = res.sort_values(by='ed')
    print(res.sort_values(by='ed').to_latex())
    '''
    datasets = ['goodreads', 'amazon_music']
    for dataset in datasets:
        up = pd.read_csv(f'{dataset}_utopia_point.tsv', sep='\t')
        print(f"{dataset}: {up['APLT_up'].mean()}")
    '''
    datasets = ['amazon_music']
    for dataset in datasets:
        up = pd.read_csv(f'{dataset}_utopia_point.tsv', sep='\t')
        files = glob.glob(f'data/population/{dataset}/*')
        manager = []
        for file in files:
            df = pd.read_csv(file, sep='\t')
            df = pd.concat([df, up], axis=1)
            df['APLT'] = (df['APLT'] - df['APLT'].min()) / (df['APLT'].max() - df['APLT'].min())
            df['Recall'] = (df['Recall'] - df['Recall'].min()) / (df['Recall'].max() - df['Recall'].min())
            df['mae'] = abs(df['APLT_up'] - df['APLT'])
            # df['ed'] = ((df['APLT_up'] - df['APLT']) ** 2 + (df['Recall_up'] - df['Recall']) ** 2) ** (1/2)
            df['APLT_d'] = (1 - df['APLT']) ** 2
            df['Recall_d'] = (1 - df['Recall']) ** 2
            df['sum_d'] = df['APLT_d'] + df['Recall_d']
            df['sum_d'] = df['sum_d'] ** 0.5
            manager.append([file, df['mae'].var(), df['sum_d'].mean()])
        res = pd.DataFrame(manager, columns=['model', 'var', 'ed'])
        res = res.sort_values(by='var')
        print(res.sort_values(by='var').to_latex())
    '''