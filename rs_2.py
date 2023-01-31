from Pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce

if __name__ == '__main__':
    model = pd.read_csv('data/EASER.tsv', sep='\t')
    obj1 = 'Recall'
    obj2 = 'APLT'
    obj3 = ''
    population = 18070
    utopia_point = np.array([1, 1])
    reference_point = np.array([0, 0])
    weights = np.array([0.5, 0.5])
    personalized_up = pd.read_csv('goodreads_utopia_point.tsv', sep='\t')
    scale = True
    calibration = True
    obj = ObjectivesSpace(model, {obj1: 'max', obj2: 'max'})
    print('****** OPTIMAL *****')
    print(obj.get_nondominated())
    print('****** DOMINATED *****')
    print(obj.get_dominated())
    if calibration:
        pdu = obj.pdu(personalized_up, scale, calibration)
    else:
        pdu = obj.pdu([utopia_point] * population, scale, calibration)
    hypervolumes = obj.hypervolumes(reference_point)
    knee_point = obj.knee_point(scale)
    euclidean_distance = obj.euclidean_distance(utopia_point, scale)
    weighted_mean = obj.weighted_mean(weights, scale)
    dfList = [pdu[['model', obj1, obj2, 'pdu']],
             hypervolumes[['model', 'hypervolume']],
             knee_point[['model', 'utility_based']], euclidean_distance[['model', 'euclidean_distance']],
             weighted_mean[['model', 'weighted_mean']]]
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='model'), dfList)
    print(df.sort_values(by=obj1).to_latex(index=False))

