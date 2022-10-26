#!/usr/bin/env python
# coding: utf-8

# In[248]:

# Compute Hausdorff dimension numerically, for the real symmetric 2D Schottky


import numpy as np
import random

# define disjoint disks (Schottky group) and associated functions

refinements = 0 # track number of refinements

N = 3 # number of disks
thetas = [np.pi/2, 7*np.pi/6, 11*np.pi/6] # angles of disk "centers"
theta_tilde = np.pi/6 # angle of arc contained in each disk

# define disks in terms of centers and radii

# maps point e^it in disc model to corresponding point in upper half plane model
# this is equivalent to the map phi(e^it) = -i*(e^it+1)/(e^it-1)
def phi(t):
    val = np.sin(t)/(np.cos(t)-1)
    
    return val

# obtain centers and radii by average and difference of endpoints of the half-circle under phi
centers = np.array([(phi(thetas[0]-theta_tilde/2) + phi(thetas[0]+theta_tilde/2))/2, (phi(thetas[1]-theta_tilde/2) + phi(thetas[1]+theta_tilde/2))/2, (phi(thetas[2]-theta_tilde/2) + phi(thetas[2]+theta_tilde/2))/2])
radii = np.array([abs(phi(thetas[0]-theta_tilde/2) - phi(thetas[0]+theta_tilde/2))/2, abs(phi(thetas[1]-theta_tilde/2) - phi(thetas[1]+theta_tilde/2))/2, abs(phi(thetas[2]-theta_tilde/2) - phi(thetas[2]+theta_tilde/2))/2])


# store function parameters (will not change in refinements)
funcs = np.array([[centers[0], radii[0]], [centers[1], radii[1]], [centers[2], radii[2]]])

# create list of function indices corresponding to sample points x_i (i.e. which original disk contains x_i?)
funcs_i = np.array([0, 1, 2])

# generate sample points in each disk
x = np.empty(N).astype(complex)
for i in range(N):
    theta = random.uniform(0, np.pi) # we want upper half of each disk
    radius = random.uniform(0, radii[i])
    x[i] = centers[i]+radius*(np.cos(theta)+1j*np.sin(theta))


# In[249]:


# define Mobius transformation, in terms of centers and radii
# a and r are center and radius of the disk about which the reflection is taken
# this map is its own inverse

def mob(a, r, z):
    val = a - r**2/(z-a)
        
    return val
    
    


# In[250]:


# define derivative of Mobius transformation analagously

def mob_prime(a, r, z):
    val = r**2/(z-a)**2
        
    return val
    
    


# In[259]:


# (1) for every i->j, solve for y_ij in P_i such that f_i(y_ij)=x_j

y = np.zeros((N,N)).astype(complex)
new_centers = np.empty(0)
new_radii = np.empty(0)

for i in range(N):
    for j in range(N):
        # P_i = D_r(a), f_i^-1(P_j) = D_s(b)
        a = centers[i]
        r = radii[i]
        
        if i != j: 
            ind = funcs_i[i] # get index of desired function
            b = (mob(funcs[ind,0], funcs[ind,1], centers[j]-radii[j]) + mob(funcs[ind,0], funcs[ind,1], centers[j]+radii[j]))/2
            s = abs(mob(funcs[ind,0], funcs[ind,1], centers[j]-radii[j]) - mob(funcs[ind,0], funcs[ind,1], centers[j]+radii[j]))/2

            if (a+r > b-s) and (a-r < b+s): # if i -> j
                y[i,j] = mob(funcs[ind,0], funcs[ind,1], x[j])
                
                # track sets for refinement
                new_centers = np.append(new_centers, b)
                new_radii = np.append(new_radii, s)
            # else y_ij = 0

print(new_centers)


# In[260]:


# (2) Compute transition matrix, such that T_ij = { |f_i'(y_ij)|^-1 if i -> j, 0 otherwise

T = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if y[i,j] != 0: # i -> j
            ind = funcs_i[i]
            T[i,j] = 1/abs(mob_prime(funcs[ind,0], funcs[ind,1], y[i,j]))
    
print(T)


# In[261]:


# (3) Solve lambda(T^alpha)=1 (elementwise exponentiation) for this partition
# This version implements the midpoint method



tol = 0.0001

# initial alpha values, such that lambda(T^alphal) < 1 < lambda(T^alpha2)
alpha1 = 10
alpha2 = 0.01
lmbd_1 = max(np.abs(np.linalg.eigvals(T**alpha1)))
lmbd_2 = max(np.abs(np.linalg.eigvals(T**alpha2)))

print(lmbd_1, lmbd_2)

alpha_mid = (alpha1+alpha2)/2
iterations = 1
num_iter = 100
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


# In[265]:


# (5) Replace P with refinement R(P), define new sample points x_ij = y_ij, return to (1)

new_x = np.empty(0).astype(complex)
new_funcs_i = np.empty(0)

for i in range(N):
    for j in range(N):
        if y[i,j] != 0: # i -> j
            new_x = np.append(new_x, y[i,j])
            new_funcs_i = np.append(new_funcs_i, funcs_i[i])


x = new_x # x_ij = y_ij
funcs_i = new_funcs_i.astype(int) 
N = x.size

# refine sets in partition
centers = new_centers
radii = new_radii

# track refinements
refinements = refinements + 1

print(x)


# In[268]:


# Print results

print('Our approximation to the dimension is ', alpha_P, '.')
print('This estimate was obtained in', refinements, 'refinements.')

asymp_est = np.log(2)/(np.log(12)-2*np.log(theta_tilde))

print('The approximation given by the asymptotic formula is ', asymp_est)


# In[ ]:




