import numpy as np
import matplotlib.pyplot as plt
def Binary_update(state, weights, step_by_step=True):
    if step_by_step==True:
        idx=np.random.randint(low=0, high=state.shape[0])
        state[idx]=np.sign(weights[idx,:]@state)
def Self_modeling(state, alpha):
    new_weights=state@state.T
    #new_weights=new_weights/(new_weights.shape[0]**2)
    new_weights=new_weights*alpha
    np.fill_diagonal(new_weights, 0)
    return new_weights
def Generate_state(size):
    state=np.random.randint(0,2, size=(size,1))
    state=np.heaviside(state, -1)
    return state
##For goof peformance normalize the weights
def Generate_weights(size, symmetrical=True):
    b=np.random.uniform(-2, 2, size=(size, size))
    b=b/(b.shape[0]*2)
    if symmetrical==True:
        b=(b+b.T)
    np.fill_diagonal(b, 0)
    return b
def Binary_energy(state, weights):
    return np.squeeze(-0.5*(state.T@weights@state))
def fitness(points, C_1=1, C_2=1):
    return np.mean(points)*C_1+np.var(points)*C_2
def zero_mut(matrix):
    ind, ind_2=np.random.randint(0, matrix.shape[0], size=(2,))
    matrix[ind, ind_2]=0
    matrix[ind_2, ind]=0
def Run_once(state, weights, num_steps, self_modeling=False):
    energy=[]
    true=[]
    if self_modeling==True:
        weights_original=np.copy(weights)
    for i in range(num_steps):
        Binary_update(state, weights)
        e=Binary_energy(state, weights)
        energy.append(e)
        if self_modeling==True:
            true_energy=Binary_energy(state, weights_original)
            true.append(true_energy)
        if self_modeling==True:
            w=Self_modeling(state, weights)
            weights+=w
    return energy,true
def get_sparsity(matrix):
    non_zero=np.count_nonzero(matrix)
    return 1-(non_zero/matrix.size)
def sparse_symm(sparsity_l,N,matrix=None):
    if matrix==None:
        conn_matrix=Generate_weights(N)
    else:
        conn_matrix=matrix
    sp=get_sparsity(conn_matrix)
    while sp<=sparsity_l:
    #for i in range(100):
        zero_mut(conn_matrix)
        sp=get_sparsity(conn_matrix)
        #print(sparsity(conn_matrix))
    else:
        #print(sparsity(conn_matrix))
        return conn_matrix
def Training_self_modeling(weights_1, relaxations, N=100, lr=0.3):
    end_list_1=[]
    weights=weights_1
    input_1=Generate_state(N)
    weights_original=np.copy(weights)
    for i in range(relaxations*10000):
    #for i in tqdm(range(relaxations*10000)):
        energy=Binary_energy(input_1, weights)
        Binary_update(input_1, weights)
        if i%10000==0 and i!=0:
            delta_w=Self_modeling(input_1, lr)
            weights+=delta_w
            weights=weights/(weights.shape[0]*2)
            true_energy=Binary_energy(input_1,weights_original)
            end_list_1.append(true_energy)
            input_1=Generate_state(N)
    weights_learned=np.copy(weights)
    return weights_learned, weights_original, end_list_1
def Measure(weights_1, weights_2, num_it=10000):
    res_dist=[]
    energy_list=[]
    for i in range(100):
        input_1=Generate_state(100)
        for j in range(num_it):
            Binary_update(input_1, weights_1)
            energy=Binary_energy(input_1, weights_2)
            energy_list.append(energy)
        res_dist.append(energy)
    return res_dist, energy_list
def augmented_energy(state, weights, cor):
    return np.squeeze(-0.5*(state.T@(weights*cor)@state))
    
