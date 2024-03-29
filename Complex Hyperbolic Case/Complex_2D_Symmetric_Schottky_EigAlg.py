# The Eigenvalue Algorithm, implemented on the symmetric Schottky group in complex hyperbolic space

import numpy as np
import random
import matplotlib.pyplot as plt


# Problem setup:


# define Schottky group of Cygan spheres, and associated complex reflections

num_refinements = 3 # number of refinements to take
refinements = 0 # track number of refinements
N = 3 # number of chains/isometric spheres

# angle of the Schottky group
theta = np.pi/100


# define the symmetric Schottky group
radii = np.array([np.tan(theta), np.tan(theta), np.tan(theta)]).astype(complex)
centers = np.array([[1 / np.cos(theta), 0], [np.exp(1j*2*np.pi/3) / np.cos(theta), 0], [np.exp(1j*4*np.pi/3) / np.cos(theta), 0]]).astype(complex)



# track centers and radii from all iterations
cumul_radii = [radii]
cumul_centers = [centers]



# store function parameters (will not change in refinements)
funcs = np.array([[radii[0], centers[0,0], centers[0,1]], [radii[1], centers[1,0], centers[1,1]], [radii[2], centers[2,0], centers[2,1]]]).astype(complex)

# create list of function indices corresponding to sample points x_i (i.e. which original sphere contains x_i?)
funcs_i = np.array([0, 1, 2])



# generate random sample points x_j in each isometric sphere S_j
x = np.empty((N,2)).astype(complex)
for i in range(N):
    r = random.uniform(0, np.abs(radii[i]))
    phi = random.uniform(0, 2*np.pi)
    rho = random.uniform(0, r)
    
    
    # zeta coordinate
    x[i,0] = centers[i,0] - rho*np.exp(1j*phi)
    
    # v coordinate
    x[i,1] = centers[i,1] + 2*np.imag(np.conj(centers[i,0])*rho*np.exp(1j*phi)) - np.sqrt(r**4-rho**4)
    


# Preliminary function definitions:


# the Cygan distance between points (zeta_1,v_1) and (zeta_2,v_2), 
# using the definition of zero norm and the group operation in the Heisenberg group
def cyg_dist(zeta_1, v_1, zeta_2, v_2):
    val = np.sqrt(np.abs( np.abs(zeta_1-zeta_2)**2 + 1j*(v_1 - v_2 - 2*np.imag(np.conj(zeta_1)*zeta_2)) ))
    
    return val


# the Koranyi inversion of a point (zeta,v)
def kor_inv(zeta, v):
    val = np.empty(2).astype(complex)
    val[0] = zeta / (np.abs(zeta)**2 - 1j*v)
    val[1] = -v / (np.abs(zeta)**4 + v**2)
    
    return val


# the complex dilation by lmbd applied to the point (zeta,v)
def comp_dil(zeta, v, lmbd):
    val = np.empty(2).astype(complex)
    val[0] = lmbd * zeta
    val[1] = np.abs(lmbd)**2 * v
    
    return val


# the Heisenberg translation by (xi,t) applied to the point (zeta,v)
def h_trans(zeta, v, xi, t):
    val = np.empty(2).astype(complex)
    val[0] = zeta + xi
    val[1] = v + t + 2*np.imag(np.conj(zeta)*xi)
    
    return val


# the complex inversion through the Cygan sphere of center (xi,t) and radius lmbd, applied to the point (zeta,v)
def comp_inv(zeta, v, lmbd, xi, t):
    tdid_val = h_trans(zeta, v, -xi, -t)
    tdi_val = comp_dil(tdid_val[0], tdid_val[1], lmbd**(-1))
    td_val = kor_inv(tdi_val[0], tdi_val[1])
    t_val = comp_dil(td_val[0], td_val[1], lmbd)
    val = h_trans(t_val[0], t_val[1], xi, t)
    
    return val


# the (square rooted) magnitude of the Jacobian determinant of a complex inversion
# serves the same purpose for this algorithm as f' did in the real case
def abs_det_comp_inv(zeta, v, lmbd, xi, t):
    val = np.abs(lmbd)**4 / cyg_dist(zeta, v, xi, t)**4
    
    return val


# determines the parameters of a chain after a complex inversion is taken
# inversion with translation parameters (xi,t) and dilation parameter lmbd is applied to the chain with center (mu,x) and complex radius rho
# the resulting chain has center (beta,gamma) and complex radius alpha
def chain_finder(lmbd, xi, t, rho, mu, x):
    val = np.empty(3).astype(complex)
    
    if abs(mu-xi) < np.finfo(complex).eps: # check mu=xi up to machine precision
        alpha = np.abs(lmbd)**2 * rho / (np.abs(rho)**2 - 1j*(x - t))
        beta = xi
        gamma = t - np.abs(lmbd)**4 * (x - t) / np.abs(np.abs(rho)**2 - 1j*(x - t))**2
    else: # non-degenerate case, proceed as normal
        alpha = np.abs(lmbd)**2 * np.conj(rho) * (mu-xi)**2 / np.abs(mu-xi)**2 \
                              / (np.abs(rho)**2 - np.abs(mu-xi)**2 + 1j*(x - t - 2*np.imag(np.conj(mu)*xi)))
        beta = xi - np.abs(lmbd)**2*(mu-xi) / (np.abs(rho)**2 - np.abs(mu-xi)**2 + 1j*(x - t - 2*np.imag(np.conj(mu)*xi)))
        gamma = t + 2*np.imag(np.abs(lmbd)**2 * np.conj(xi) * (mu-xi) \
                              / (np.abs(rho)**2 - np.abs(mu-xi)**2 + 1j*(-2*np.imag(np.conj(mu)*xi) + x - t))) \
                              - np.abs(lmbd)**4*(x - t - 2*np.imag(np.conj(mu)*xi)) \
                              / np.abs(np.abs(rho)**2 - np.abs(mu-xi)**2 + 1j*(x - t - 2*np.imag(np.conj(mu)*xi)))**2

    val[0] = alpha
    val[1] = beta
    val[2] = gamma
    return val



# Test 1
# for disjoint Cygan spheres determined by the parameters below, should print 'True'
# numerically tests that the chainfinder function actually represents an involution

#lmbd, xi, t = 1.5*np.exp(1j*2), -3+1j, 2
#rho, mu, xx = 0.5*np.exp(1j*-4), 2-1j, -1
#alpha, beta, gamma = chain_finder(lmbd, xi, t, rho, mu, xx)

#tol = 1e-10

#print(np.linalg.norm(chain_finder(lmbd, xi, t, alpha, beta, gamma) - np.array([rho, mu, xx])) < tol)

# Test 2
# should print 'True'
# ensures that the random sample points lie in their respective Cygan spheres

#isin = True
#for i in range(N):
    #isin = isin and cyg_dist(x[i,0], x[i,1], centers[i,0], centers[i,1]) < np.abs(radii[i])
    
    #if isin == False:
        #break
    
#print(isin)



# The Eigenvalue Algorithm:


while refinements < num_refinements:

  # (1) for every i->j, solve for y_ij in isometric sphere S_i such that f_i(y_ij)=x_j

  y = np.zeros((N,N,2)).astype(complex)
  new_centers = np.empty((0,2)).astype(complex)
  new_radii = np.empty(0).astype(complex)

  for i in range(N):
      for j in range(N):
          # get radius and center for i-th chain/isometric sphere
          lmbd, xi, t = radii[i], centers[i,0], centers[i,1]

          # get radius and center for j-th chain/isometric sphere
          rho, mu, xx = radii[j], centers[j,0], centers[j,1]

          if i != j: 
              ind = funcs_i[i] # get index of desired function

              # get radius and center of chain under inverse complex reflection
              # complex reflections are involutions, so each is its own inverse
              alpha, beta, gamma = chain_finder(funcs[ind,0], funcs[ind,1], funcs[ind,2], rho, mu, xx)

              if cyg_dist(xi, t, beta, gamma) < np.abs(lmbd) - np.abs(alpha): # if i -> j
                  y[i,j,:] = comp_inv(x[j,0], x[j,1], funcs[ind,0], funcs[ind,1], funcs[ind,2])

                  # track sets for refinement
                  new_radii = np.append(new_radii, alpha)
                  new_centers = np.append(new_centers, [[beta, gamma]], axis=0)

              # else y_ij = 0





  # (2) Compute transition matrix, such that T_ij = { |f_i'(y_ij)|^-1 if i -> j, 0 otherwise

  T = np.zeros((N,N))

  for i in range(N):
      for j in range(N):
          if y[i,j,:].any() != 0: # i -> j
              ind = funcs_i[i]
              T[i,j] = 1/abs_det_comp_inv(y[i,j,0], y[i,j,1], funcs[ind,0], funcs[ind,1], funcs[ind,2])






  # (3) Solve lambda(T^alpha)=1 (elementwise exponentiation) for this partition
  # This implementation uses the Bisection Method



  tol = 0.0001

  # initial alpha values, such that lambda(T^alphal) < 1 < lambda(T^alpha2)
  # the particular values used below are not guaranteed to work, but they have in every test case so far
  alpha1 = 1
  alpha2 = 0.01
  lmbd_1 = max(np.abs(np.linalg.eigvals(T**alpha1)))
  lmbd_2 = max(np.abs(np.linalg.eigvals(T**alpha2)))



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





  # (4) Output alpha

  alpha_P = alpha_mid




  # (5) Replace P with refinement R(P), define new sample points x_ij = y_ij, return to (1)

  new_x = np.empty((0,2)).astype(complex)
  new_funcs_i = np.empty(0)

  for i in range(N):
      for j in range(N):
          if y[i,j,0:].any() != 0: # i -> j
              new_x = np.append(new_x, [y[i,j,:]], axis=0)
              new_funcs_i = np.append(new_funcs_i, funcs_i[i])


  x = new_x # x_ij = y_ij
  funcs_i = new_funcs_i.astype(int) 
  N = x.shape[0]

  # refine sets in partition
  centers = new_centers
  radii = new_radii
  cumul_centers.append(centers)
  cumul_radii.append(radii)

  # track refinements
  refinements = refinements + 1






# Test 3
# should print 'True'
# ensures that the iterated sample points lie in their respective Cygan spheres

#isin = True
#for i in range(N):
    #isin = isin and cyg_dist(x[i,0], x[i,1], centers[i,0], centers[i,1]) < np.abs(radii[i])
    #print(cyg_dist(x[i,0], x[i,1], centers[i,0], centers[i,1])) 
    #print(np.abs(radii[i]))
    #if isin == False:
        #break
    
#print(isin)



# Results:


print('computed dimension, after', refinements, 'refinements:', alpha_P)
print('asymptotic dimension approximation:', np.log(2)/(np.log(12)-4*np.log(np.sin(theta))))




# plotting the spheres
# this implementation plots 2M^2 "evenly" spaced points on each sphere
ax = plt.axes(projection = '3d')

# specify bounds, adjust as needed
L = 2
ax.set(xlim=(-L,L), ylim=(-L,L), zlim=(-L,L))

# add axis labels
plt.rcParams['text.usetex'] = True
ax.set_xlabel('$\mathrm{Re}(\zeta)$')
ax.set_ylabel('$\mathrm{Im}(\zeta)$')
ax.set_zlabel('$v$')
ax.set_title('')

# 2M^2 points on each sphere
M = 50

# colors to use
cs = ('red', 'yellow', 'blue')
num_cs = 3

for i in range(num_refinements+1):
    cens = cumul_centers[i]
    rads = cumul_radii[i]
    for j in range(cens.shape[0]):
        cen = cens[j,:]
        rad = rads[j]

        # linearly space points on Cygan spheres
        phi = np.linspace(0, 2*np.pi, M)
        rho = np.linspace(0, rad, M)
        
        # zeta coordinate
        zetas = cen[0]*np.ones((M,M)) - np.outer(rho, np.exp(1j*phi))

        # v coordinate
        vs = cen[1]*np.ones((M,M)) + 2*np.imag(np.conj(cen[0])*np.outer(rho, np.exp(1j*phi))) \
             - np.sqrt(rad**4*np.ones((M,M))-np.transpose(np.tile(rho**4, (M,1))))

        # flatten arrays of points and account for upper and lower half of spheres
        zetas = zetas.flatten()
        zetas = np.tile(zetas, (1,2))
        neg_vs = vs + 2*np.sqrt(rad**4*np.ones((M,M))-np.transpose(np.tile(rho**4, (M,1))))
        vs = np.concatenate((vs, neg_vs))
        vs = vs.flatten()

        # plot the points
        ax.scatter3D(np.real(zetas), np.imag(zetas), np.real(vs), marker = '.', c = cs[np.mod(i,num_cs)], alpha = 0.02)
    
    # for 2 refinements, helps visually, but there is probably a better solution
    M = M//2
        



