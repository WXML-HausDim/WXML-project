#!/usr/bin/env python
# coding: utf-8

# In[203]:


import numpy as np
import random

# define disjoint disks (Schottky group) and associated functions

N = 2 # number of disks

# define disks in terms of centers and radii
centers = np.array([-2, 2]).astype(complex)
radii = np.array([1, 1]).astype(complex)


# create list of disk/domain indices corresponding to sample points x_i (i.e. which original disk contains x_i?)
disks = np.array([0, 1])

# generate sample points in each disk
x = np.empty(N).astype(complex)
for i in range(N):
    theta = random.uniform(0, np.pi) # we want upper half of disk
    radius = random.uniform(0, radii[i])
    x[i] = centers[i]+radius*(np.cos(theta)+1j*np.sin(theta))



# In[204]:


# define Mobius transformation, in terms of centers and radii
# a and r are center and radius of the disk about which the reflection is taken
# this map is its own inverse

def mob(a, r, z):
    val = a - r**2/(z-a)
        
    return val
    
    


# In[205]:


# define derivative of Mobius transformation analagously

def mob_prime(a, r, z):
    val = r**2/(z-a)**2
        
    return val
    
    


# In[206]:


# (1) for every i->j, solve for y_ij in P_i such that f_i(y_ij)=x_j

y = np.zeros((N,N)).astype(complex)


for i in range(N):
    for j in range(N):
        # we want inverse mobius transform, which is the same as the transform itself
        if i != j:
            ind = disks[i]
            y[i,j] = mob(centers[ind], radii[ind], x[j])

#print(y)


# In[207]:


# (2) Compute transition matrix, such that T_ij = { |f_i'(y_ij)|^-1 if i -> j, 0 otherwise

T = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i != j:
            ind = disks[i]
            T[i,j] = 1/abs(mob_prime(centers[ind], radii[ind], y[i,j]))
    
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

for i in range(N):
    for j in range(N):
        if y[i,j] != 0:
            new_x = np.append(new_x, y[i,j])
            new_disks = np.append(new_disks, disks[i])


x = new_x
disks = new_disks.astype(int)
N = x.size

# Notes:
# originally checked i -> j with condition "if not ( [i,j] in pairs.tolist() )"
# this does not work after the first iteration because i and j no longer represent the ith and jth disks
# now check using "if not ( [doms[i],doms[j]] in pairs.tolist() )"
# doms exists to check i -> j
# the problem with this is that the blocks in the refinement are no longer the whole first disk of the mapping,
# so not guaranteed that anything outside of the second disk will have positive interssection with image

# this note ^^^ is from the pre 4-15 version, the condition is now "if i != j", and this has the same issue

print(x)


# In[ ]:





# In[ ]:




