from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianModel
import pandas as pd


def bif_to_adj_matrix(file_path):
    reader = BIFReader(file_path)
    model = reader.get_model()
    bayesian_model = BayesianModel(model.edges())
    nodes = bayesian_model.nodes()
    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    for edge in bayesian_model.edges():
        adj_matrix[edge[1]][edge[0]] = 1

    return adj_matrix



bif_files = ['asia.bif', 'cancer.bif', 'earthquake.bif', 'sachs.bif', 'survey.bif'] # Small Networks (<20 nodes)
# bif_files = ['alarm.bif', 'barley.bif', 'child.bif', 'insurance.bif', 'mildew.bif', 'water.bif']  # Medium Networks (20–50 nodes)
# bif_files = ['hailfinder.bif', 'hepar2.bif', 'win95pts.bif'] # Large Networks (50–100 nodes)
# bif_files = ['andes.bif', 'diabetes.bif', 'link.bif', 'munin1.bif', 'pathfinder.bif', 'pigs.bif'] # Very Large Networks (100–1000 nodes)
# bif_files = ['munin.bif', 'munin2.bif', 'munin3.bif', 'munin4.bif'] # Massive Networks (1000+ nodes)
for bif_file in bif_files:
    adj_matrix = bif_to_adj_matrix(bif_file)
    csv_file = bif_file.replace('.bif', '.csv')
    print(adj_matrix)
    adj_matrix.to_csv(f'label/{csv_file}', index=False)

