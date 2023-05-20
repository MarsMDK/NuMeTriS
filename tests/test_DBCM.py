import numpy as np
import sys
import unittest

sys.path.append("../src/")
import NuMeTriS as nm




class test_DBCM(unittest.TestCase):
    
    
    
    def test_DBCM(self):

        path_tradeflow = 'tradeflow_'+str(2000)+'.npy'
        path_nodes = 'nodes_'+str(2000)+'.npy'
        tradeflow = np.load(path_tradeflow,allow_pickle=True)
        nodes = np.load(path_nodes,allow_pickle=True)
        nodes_idx = np.arange(len(nodes))
        G = nm.Graph(adjacency=tradeflow)
        G.solver(model='DBCM')    
        G.numerical_triadic_zscores()
        G.plot_zscores()
        self.assertTrue(G.norm < 1e-05,'solution reached!')
    
    
if __name__ == '__main__':
    unittest.main()