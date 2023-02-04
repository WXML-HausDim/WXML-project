#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import random

# symmetric theta-Schottky, implemented in the disk model
# here phi represents theta_tilde used in previous convention

# define disjoint disks (Schottky group) and associated functions

refinements = 0 # track number of refinements
N = 3 # number of disks

# if converting from disk model to half-plane model

thetas = [np.pi/3, np.pi, 5*np.pi/3] # angles of disk "centers" # angle of each center (central angle)
phi = 2*np.pi/3.1
phis = [phi, phi, phi] # angle of arc contained in each disk (arc angle)


# for drawing
data = np.array([thetas, phis])

# track centers and radii from all iterations
cumul_thetas = [thetas]
cumul_phis = [phis]

# store function parameters (will not change in refinements)
funcs = np.array([[thetas[0], phis[0]], [thetas[1], phis[1]], [thetas[2], phis[2]]])

# create list of function indices corresponding to sample points x_i (i.e. which original disk contains x_i?)
funcs_i = np.array([0, 1, 2])

# generate sample points in each disk
# not implemented randomly: takes midpoint of endpoints as sample point
x = np.empty(N).astype(complex)
for i in range(N):
    x[i] = (np.exp(1j*(thetas[i]+phis[i]/2)) + np.exp(1j*(thetas[i]-phis[i]/2)))/2

print(data)


# In[5]:


# define Mobius transformation, in terms of disk parameters
# theta and phi are the central angle and arc angle of the disk through which the reflection is taken
# this map is its own inverse, i.e. an  involution

def mob(theta, phi, z):
    u = 1j/np.sin(phi/2)
    v = np.sin(theta)/np.tan(phi/2) + 1j*np.cos(theta)/np.tan(phi/2)
    val = (u*z + np.conj(v)) / (v*z + np.conj(u))
        
    return val

# define derivative of Mobius transformation analagously

def mob_prime(theta, phi, z):
    u = 1j/np.sin(phi/2)
    v = np.sin(theta)/np.tan(phi/2) + 1j*np.cos(theta)/np.tan(phi/2)
    val = 1/(v*z + np.conj(u))**2
        
    return val
    
# compute angle of a complex number z, in [0,2pi]
def angle(z):
    ang = np.angle(z)
    val = ang + 2*np.pi*(ang < 0)
    
    return val


# In[36]:


# (1) for every i->j, solve for y_ij in P_i such that f_i(y_ij)=x_j

y = np.zeros((N,N)).astype(complex)
new_thetas = np.empty(0)
new_phis = np.empty(0)

for i in range(N):
    for j in range(N):
        # P_i = D_r(a), f_i^-1(P_j) = D_s(b)
        th = thetas[i]
        ph = phis[i]
        
        if i != j: 
            ind = funcs_i[i] # get index of desired function
            th_j = (angle(mob(funcs[ind,0], funcs[ind,1], np.exp(1j*(thetas[j]-phis[j]/2)))) +                     angle(mob(funcs[ind,0], funcs[ind,1], np.exp(1j*(thetas[j]+phis[j]/2)))))/2
            ph_j = abs(angle(mob(funcs[ind,0], funcs[ind,1], np.exp(1j*(thetas[j]-phis[j]/2)))) -                     angle(mob(funcs[ind,0], funcs[ind,1], np.exp(1j*(thetas[j]+phis[j]/2)))))
            
            #print(th, th_j)
            #print(ph, ph_j)
            #print('\n')

            if (th+ph/2 > th_j-ph_j/2) and (th-ph/2 < th_j+ph_j/2): # if i -> j
                y[i,j] = mob(funcs[ind,0], funcs[ind,1], x[j])

                # track sets for refinement
                new_thetas = np.append(new_thetas, th_j)
                new_phis = np.append(new_phis, ph_j)
                # else y_ij = 0

#print(new_thetas)


# In[37]:


# (2) Compute transition matrix, such that T_ij = { |f_i'(y_ij)|^-1 if i -> j, 0 otherwise

T = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if y[i,j] != 0: # i -> j
            ind = funcs_i[i]
            T[i,j] = 1/abs(mob_prime(funcs[ind,0], funcs[ind,1], y[i,j]))
    
#print(T)


# In[38]:


# (3) Solve lambda(T^alpha)=1 (elementwise exponentiation) for this partition
# This version implements the midpoint method



tol = 0.0001

# initial alpha values, such that lambda(T^alphal) < 1 < lambda(T^alpha2)
alpha1 = 1
alpha2 = 0.01

lmbd_1 = max(np.abs(np.linalg.eigvals(T**alpha1)))
lmbd_2 = max(np.abs(np.linalg.eigvals(T**alpha2)))

print(lmbd_1, lmbd_2)

while (lmbd_1-1)*(lmbd_2-1)>0:
    alpha1 = alpha1*2
    alpha2 = alpha2/2
    
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


# In[39]:


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
thetas = new_thetas
phis = new_phis
cumul_thetas.append(thetas)
cumul_phis.append(phis)
data = np.append(data, [thetas, phis], axis=1)

# track refinements
refinements = refinements + 1

#print(data)


# In[40]:


# Print results

print('Our approximation to the dimension is ', alpha_P, '.')
print('This estimate was obtained in', refinements, 'refinements.')

asymp_est = np.log(2)/(np.log(12)-2*np.log(phi))

print('The approximation given by the asymptotic formula is ', asymp_est)


# In[41]:


#np.savetxt('symmdisk7.csv', data, delimiter=',')


# In[ ]:





# In[ ]:




