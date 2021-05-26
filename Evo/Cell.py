import numpy as np
from Evo.Hebbian import Hebbian_update
from Evo.Basic_activations import *
class Cell(object):
    def __init__(self, genes, EXSPRESSION_MATRIX=None, REGULATION_MATRIX=None, mutation_rate=0.01, id=0, alpha=0.01, Rand_Reg=True):
        #self.EXSPRESSION_MATRIX=np.random.rand(number_of_genes,1)
        #self.REGULATION_MATRIX=np.zeros((number_of_genes,number_of_genes))
        self.EXSPRESSION_MATRIX=EXSPRESSION_MATRIX
        self.genes=genes
        self.alpha=alpha
        self.id=0
        self.Rand_Reg=Rand_Reg
        self.REGULATION_MATRIX=REGULATION_MATRIX
        self.mr=mutation_rate
        if self.EXSPRESSION_MATRIX is None:
            self.EXSPRESSION_MATRIX=np.random.uniform(-1,1, [genes,1])
        if self.REGULATION_MATRIX is None:
            if self.Rand_Reg==True:
                self.REGULATION_MATRIX=np.random.normal(0.0, 0.1, [self.genes, self.genes])
            if self.Rand_Reg==False:
                self.REGULATION_MATRIX=np.zeros((self.genes,self.genes))
        self.Energy=None
        self.Size=None
    def Calculate_exspression(self):
        #self.EXSPRESSION_MATRIX=self.EXSPRESSION_MATRIX+0.001*(np.matmul(self.REGULATION_MATRIX, tanh(self.EXSPRESSION_MATRIX))-self.EXSPRESSION_MATRIX)
        self.EXSPRESSION_MATRIX=self.EXSPRESSION_MATRIX+0.001*(np.matmul(self.REGULATION_MATRIX, tanh(self.EXSPRESSION_MATRIX)))
    def Basic_mutation(self):
        self.EXSPRESSION_MATRIX[np.random.randint(0, self.EXSPRESSION_MATRIX.shape[0])]+=np.random.uniform(-0.1,0.1)
    def Regulation_update(self):
        weights_update=Hebbian_update(self.alpha, self.EXSPRESSION_MATRIX)
        self.REGULATION_MATRIX=self.REGULATION_MATRIX+weights_update
    def Regulation_mutation(self, num_mutations=1):
        for m in range(num_mutations):
            cor=np.random.randint(0, high=self.EXSPRESSION_MATRIX.shape[0], size=2)
            i,j=cor[0],cor[1]
            self.REGULATION_MATRIX[i][j]+=np.random.uniform(-0.1,0.1)
    def End_life(self):
        self.EXSPRESSION_MATRIX=None
        self.REGULATION_MATRIX=None
    def Make_offspring(self, gen_size=10, inherit_regulation=True):
        if inherit_regulation==True:
            self.generation=[Cell(self.genes, np.copy(self.EXSPRESSION_MATRIX), np.copy(self.REGULATION_MATRIX)) for i in range(gen_size)]
        if inherit_regulation==False:
            self.generation=[Cell(self.genes, np.copy(self.EXSPRESSION_MATRIX)) for i in range(gen_size)]
        #for i in range(10):
             #New_cell=Cell(self.genes, EXSPRESSION_MATRIX=np.copy(self.EXSPRESSION_MATRIX), REGULATION_MATRIX=self.REGULATION_MATRIX, id=i)
             #self.generation.append(New_cell)
        return self.generation[:]
        