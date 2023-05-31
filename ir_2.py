from Pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce


def choice(x, min):
    return 1 if x == min else 0


if __name__ == '__main__':
    model = pd.read_csv('data/trees_new.tsv', sep='\t')
    obj1 = 'nDCG'
    obj2 = 'time'
    obj3 = 'energy'
    population = 6306
    utopia_point = np.array([1, 0, 0])
    reference_point = np.array([0.5, 0.00002, 0.001])
    weights = np.array([0.5, 0.5, 0.5])
    scale = False
    calibration = False
    obj = ObjectivesSpace(model, {obj1: 'max', obj2: 'min', obj3: 'min'})
    print('****** OPTIMAL *****')
    print(obj.get_nondominated())
    print('****** DOMINATED *****')
    print(obj.get_dominated())
    pdu = obj.pdu([utopia_point] * population, 'msn', scale, calibration)
    hypervolumes = obj.hypervolumes(reference_point)
    knee_point = obj.knee_point(scale)
    euclidean_distance = obj.euclidean_distance(utopia_point, scale)
    weighted_mean = obj.weighted_mean(weights, scale)
    dfList = [pdu[['model', obj1, obj2, obj3, 'pdu']],
                  hypervolumes[['model', 'hypervolume']],
                  knee_point[['model', 'utility_based']], euclidean_distance[['model', 'euclidean_distance']],
                  weighted_mean[['model', 'weighted_mean']]]
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='model'), dfList)
    print(df.sort_values(by=obj1).to_latex(index=False))
    df['choicepdu'] = df['pdu'].map(lambda x: choice(x, df['pdu'].min()))
    df['choicehv'] = df['hypervolume'].map(lambda x: choice(x, df['hypervolume'].max()))
    df['choiceub'] = df['utility_based'].map(lambda x: choice(x, df['utility_based'].max()))
    df['choiceed'] = df['euclidean_distance'].map(lambda x: choice(x, df['euclidean_distance'].min()))
    df['choicewm'] = df['weighted_mean'].map(lambda x: choice(x, df['weighted_mean'].max()))
    df['choiceno'] = df['choicepdu'] + df['choicehv'] + df['choiceub'] + df['choiceed'] + df['choicewm']
    df['time'] = df['time'] * 10e6
    print(df.sort_values(by=obj1).to_latex(index=False))
    # df.to_csv('to_latex/treesnew3d.csv', sep=',', index=False)



