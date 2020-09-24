# taken from 
# https://dataorigami.net/blogs/napkin-folding/79031811-multi-armed-bandits

from scipy.stats.distributions import beta as sbeta
import numpy as np

rand = np.random.rand



class Bandits(object):
    """
    This class represents N bandits machines.
    parameters:
        p_array: a (n,) Numpy array of probabilities >0, <1.
    methods:
        pull( i ): return the results, 0 or 1, of pulling 
                   the ith bandit.
    """
    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)
        
    def pull( self, i ):
        #i is which arm to pull
        return rand() < self.p[i]
    
    def __len__(self):
        return len(self.p)

    
    
class BayesianStrategy( object ):
    """
    Implements a online, learning strategy to solve
    the Multi-Armed Bandit problem.
    
    parameters:
        bandits: a Bandit class with .pull method
    
    methods:
        sample_bandits(n): sample and train on n pulls.
    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    """
    
    def __init__(self, bandits):
        
        self.bandits = bandits
        n_bandits = len( self.bandits )
        self.wins = np.zeros( n_bandits )
        self.trials = np.zeros(n_bandits )
        self.N = 0
        self.choices = []
        self.bb_score = []

    
    def sample_bandits( self, n=1 ):
        
        bb_score = np.zeros( n )
        choices = np.zeros( n )
        
        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            choice = np.argmax( rbeta( 1 + self.wins, 1 + self.trials - self.wins) )
            print(choice)
            #sample the chosen bandit
            result = self.bandits.pull( choice )
            print(result)
            
            #update priors and score
            self.wins[ choice ] += result
            self.trials[ choice ] += 1
            bb_score[ k ] = result 
            self.N += 1
            choices[ k ] = choice
            
        self.bb_score = np.r_[ self.bb_score, bb_score ]
        self.choices = np.r_[ self.choices, choices ]
        return 
    

    
def rbeta(alpha, beta, size=None):
    """
    Random beta variates.
    """
    return sbeta.ppf(np.random.random(size), alpha, beta)