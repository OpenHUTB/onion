import brainscore
import os
import pickle

filename = 'dicarlo.MajajHong2015.public'
# filename = 'dicarlo.Majaj2015'

saved_filename = os.path.join('data', filename)
# if os.path.exists(saved_filename) == False:
#     os.mkdir('data')

if os.path.exists(saved_filename) == False:
    data = brainscore.get_assembly(filename)
    with open(saved_filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(saved_filename, 'rb') as f:
        data = pickle.load(f)


from brainscore.metrics.rdm import RDM
metric = RDM()
score = metric(assembly1=data, assembly2=data)  # np.corrcoef
# > Score(aggregation: 2)>
# > array([1., 0.])
# > Coordinates:
# >   * aggregation    'center' 'error'
