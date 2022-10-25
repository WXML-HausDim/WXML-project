#!/usr/bin/env python
# coding: utf-8

# In[203]:


import numpy as np
import random

# define disjoint disks (Schottky group) and associated functions

N = 6 # number of disks

# define disks in terms of centers and radii
centers = np.array([-1, 1, -4, 4, -7, 7]).astype(complex)
radii = np.array([1, 1, 1, 1, 1, 1]).astype(complex)


# create list of indices of paired disks
pairs = np.array([[0,1], [1,0], [2,3], [3,2], [4,5], [5,4]])

# create list of disk/domain indices corresponding to sample points x_i (i.e. which original disk contains x_i?)
disks = np.array([0, 1, 2, 3, 4, 5])

# generate sample points in each disk
x = np.empty(N).astype(complex)
for i in range(N):
    theta = random.uniform(0, 2*np.pi)
    radius = random.uniform(0, radii[i])
    x[i] = centers[i]+radius*(np.cos(theta)+1j*np.sin(theta))



# In[204]:


# define Mobius transformation, in terms of centers and radii
# a and r are center and radius of first disk, b and s center and radius of second

def mob(a,r, b,s, z):
    val = b - s*r/(z-a)
        
    return val
    
    


# In[205]:


# define derivative of Mobius transformation analagously

def mob_prime(a,r, b,s, z):
    val = s*r/(z-a)**2
        
    return val
    
    


# In[206]:


# (1) for every i->j, solve for y_ij in P_i such that f_i(y_ij)=x_j

y = np.zeros((N,N)).astype(complex)


for i in range(N):
    for j in range(N):
        # we want inverse mobius transform, which we can find just by reversing order of parameters
        if not ( [disks[i],disks[j]] in pairs.tolist() ):
            ind = pairs[i]
            y[i,j] = mob(centers[ind[1]], radii[ind[1]], centers[ind[0]], radii[ind[0]], x[j])

#print(y)


# In[207]:


# (2) Compute transition matrix, such that T_ij = { |f_i'(y_ij)|^-1 if i -> j, 0 otherwise

T = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if not ( [disks[i],disks[j]] in pairs.tolist() ):
            ind = pairs[i]
            T[i,j] = 1/abs(mob_prime(centers[ind[0]], radii[ind[0]], centers[ind[1]], radii[ind[1]],                                        y[i,j]))
    
#print(T)


# In[208]:


# (3) Solve lambda(T^alpha)=1 (elementwise exponentiation) for this partition
# This version implements the midpoint method



tol = 1e-15

# initial alpha values, such that lambda(T^alphal) < 1 < lambda(T^alpha2)
alpha1 = 2
alpha2 = 0.2
lmbd_1 = max(np.abs(np.linalg.eigvals(T**alpha1)))
lmbd_2 = max(np.abs(np.linalg.eigvals(T**alpha2)))

print(lmbd_1, lmbd_2)

alpha_mid = (alpha1+alpha2)/2
iterations = 1
num_iter = 1000
lmbd_mid = max(np.abs(np.linalg.eigvals(T**alpha_mid)))

while np.abs(lmbd_mid-1) > tol and iterations < num_iter:
    if (lmbd_mid-1)*(lmbd_1-1) < 0:
        alpha2 = alpha_mid
        lmbd_2 = max(np.abs(np.linalg.eigvals(T**alpha2)))
    else:
        alpha1 = alpha_mid
        lmbd_1 = max(np.abs(np.linalg.eigvals(T**alpha1)))
    
    
    alpha_mid = (alpha1+alpha2)/2
    lmbd_mid = max(np.abs(np.linalg.eigvals(T**alpha_mid)))
    iterations = iterations + 1

print(lmbd_mid, alpha_mid, alpha1, alpha2)
print(iterations)



# (4) Output alpha

alpha_P = alpha_mid


# In[209]:


# (5) Replace P with refinement R(P), define new sample points x_ij = y_ij, return to (1)

new_x = np.empty(0).astype(complex)
new_disks = np.empty(0)
new_pairs = np.empty((0,2))

for i in range(N):
    for j in range(N):
        if y[i,j] != 0:
            new_x = np.append(new_x, y[i,j])
            new_disks = np.append(new_disks, disks[i])
            new_pairs = np.append(new_pairs, pairs[i].reshape(1,2), axis=0)


x = new_x
disks = new_disks.astype(int)
pairs = new_pairs.astype(int)
N = x.size

# Notes:
# originally checked i -> j with condition "if not ( [i,j] in pairs.tolist() )"
# this does not work after the first iteration because i and j no longer represent the ith and jth disks
# now check using "if not ( [doms[i],doms[j]] in pairs.tolist() )"
# doms exists to check i -> j
# the problem with this is that the blocks in the refinement are no longer the whole first disk of the mapping,
# so not guaranteed that anything outside of the second disk will have positive interssection with image

print(x)


# In[ ]:





# In[ ]:




