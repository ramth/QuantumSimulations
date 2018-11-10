import numpy as np
import itertools
import scipy

import scipy.sparse
import scipy.sparse.linalg
import matplotlib
import matplotlib.pyplot as plt

class QPotentials():
    
    @staticmethod
    def quartic(nu):
        return lambda x : x**2 + nu*x**4
   
    @staticmethod
    def quadratic():
        return lambda x : x**2

class QSystem():

    def __init__(self,L,N,V_func):
        self.L = L
        self.N = N
        self.X = np.linspace(-0.5*L,0.5*L,N)

        #Generate matrices
        self.K_mat = scipy.sparse.diags([1,1,1],
                                    [-1,0,1],
                                    (N,N))
        self.K_mat = self.K_mat*(size/L)**2 #2nd derivative factor
        self.V_mat = scipy.sparse.diags(V_func(self.X))
  
        #This assumes dimensionless units..
        self.H = K_mat + V_mat
        
    def find_eigenstates(self):
        self.energies, self.eigenstates = np.linalg.eigh(H.toarray())

    def plot_with_eigenstates(self,states):
        #Plot Potential and overlay scaled eigenstates on top of it
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Potential')
        ax1.plot(x_0,V_x_0)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Probability Amplitude')
        for state in states:
            ax2.plot(x_0,eigen_vecs[:,state]*np.conj(eigen_vecs[:,state]),color='r')
        plt.show()

    def evolve_psi(psi,eigen_vals,eigen_vecs):
        eigen_len = len(self.energies)
        psi_t = np.zeros(psi.shape)
        for idx in range(0,eigen_len):
            A = np.conj(self.eigenstates[:,idx])*psi*np.exp(i*self.energies[idx])
            psi_t += A
        return psi_t

def quartic():

    #Initialize parameters
    size = 1000
    L = 3
    x_0 = np.linspace(-0.5*L,0.5*L,1000)
    nu = 0.1
    V_x_0 = x_0**2 + nu*x_0**4

    #Generate matrics
    diagonals = np.ones((3,size))
    K_mat = scipy.sparse.diags(
                [1,1,1],
                [-1,0,1],
                (size,size))
    V_mat = scipy.sparse.diags(V_x_0)
    
    H = K_mat*(size/L) + V_mat
    
    eigen_vals, eigen_vecs = np.linalg.eigh(H.toarray())
   
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Potential')
    ax1.plot(x_0,V_x_0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Probability Amplitude')
    ax2.plot(x_0,eigen_vecs[:,0]*np.conj(eigen_vecs[:,0]),color='r')
    plt.show()
    
if __name__ == "__main__":
    L =3
    size = 1000
    x_0 = np.linspace(-0.5*L, 0.5*L, size)
    nu = 0.1
    qsys = (x_0, QPotentials.quartic(nu))
