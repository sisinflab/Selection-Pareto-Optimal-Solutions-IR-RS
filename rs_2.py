from Pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce


def choice(x, min):
    return 1 if x == min else 0


if __name__ == '__main__':
    model = pd.read_csv('data/EASER_amazonmusic.tsv', sep='\t')
    obj1 = 'Recall'
    obj2 = 'APLT'
    population = 14354
    utopia_point = np.array([1, 1])
    reference_point = np.array([0, 0])
    weights = np.array([0.5, 0.5])
    personalized_up = pd.read_csv('amazon_music_utopia_point.tsv', sep='\t')
    scale = True
    # calibration = False
    obj = ObjectivesSpace(model, {obj1: 'max', obj2: 'max'})
    print('****** OPTIMAL *****')
    print(obj.get_nondominated())
    print('****** DOMINATED *****')
    print(obj.get_dominated())
    # if calibration:
    c_pdu = obj.pdu(personalized_up, 'amazon_music', scale, calibration=True)
    pdu = obj.pdu([utopia_point] * population, 'amazon_music', scale, calibration=False)
    hypervolumes = obj.hypervolumes(reference_point)
    knee_point = obj.knee_point(scale)
    euclidean_distance = obj.euclidean_distance(utopia_point, scale)
    weighted_mean = obj.weighted_mean(weights, scale)
    dfList = [pdu[['model', obj1, obj2, 'pdu']],
              c_pdu[['model', 'c_pdu']],
             hypervolumes[['model', 'hypervolume']],
             knee_point[['model', 'utility_based']], euclidean_distance[['model', 'euclidean_distance']],
             weighted_mean[['model', 'weighted_mean']]]
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='model'), dfList)
    #df['Recall'] = df['Recall'].apply(lambda x: round(x, 4))
    #df['APLT'] = df['APLT'].apply(lambda x: round(x, 4))
    #df['pdu'] = df['pdu'].apply(lambda x: round(x, 4))
    #df['c_pdu'] = df['c_pdu'].apply(lambda x: round(x, 4))
    # df['hypervolume'] = df['hypervolume'] * 10e-1
    #df['hypervolume'] = df['hypervolume'].apply(lambda x: round(x, 5))
    #df['euclidean_distance'] = df['euclidean_distance'].apply(lambda x: round(x, 4))
    #df['weighted_mean'] = df['weighted_mean'].apply(lambda x: round(x, 4))

    df['choicepdu'] = df['pdu'].map(lambda x: choice(x, df['pdu'].min()))
    df['choicecpdu'] = df['c_pdu'].map(lambda x: choice(x, df['c_pdu'].min()))
    df['choicehv'] = df['hypervolume'].map(lambda x: choice(x, df['hypervolume'].max()))
    df['choiceub'] = df['utility_based'].map(lambda x: choice(x, df['utility_based'].max()))
    df['choiceed'] = df['euclidean_distance'].map(lambda x: choice(x, df['euclidean_distance'].min()))
    df['choicewm'] = df['weighted_mean'].map(lambda x: choice(x, df['weighted_mean'].max()))
    df['choiceno'] = df['choicepdu'] + df['choicehv'] + df['choiceub'] + df['choiceed'] + df['choicewm'] + df['choicecpdu']
    print(df.sort_values(by=obj1).to_latex(index=False))
    #df.sort_values(by=obj1).to_csv('to_latex/easeramazon.csv', sep=',', index=False)
