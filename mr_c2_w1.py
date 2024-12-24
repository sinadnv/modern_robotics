# Modern Robotics Course 2, Week 1 Final Quiz
# The code calculates M, Slist, Blist, and T(theta) for both s- and b- frames and shows how different methods yield same response.

import numpy as np
import modern_robotics as mr


sq3 = np.sqrt(3)
theta = np.array([-np.pi/2, np.pi/2, np.pi/3, -np.pi/4, 1, np.pi/6])
M = np.array([[1,0,0,2+sq3],[0,1,0,0],[0,0,1,1+sq3],[0,0,0,1]])                 # Question 1

# S Frame
Slist = np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],
                  [0,0,1,1-sq3,0,0],[-1,0,0,0,0,-2-sq3],[0,1,1+sq3,2+sq3,1,0]]) # Question 2

semat_s = []
for s in np.transpose(Slist):
    semat_s.append(mr.VecTose3(s))

T_s = []
i = 0 
for s in semat_s:
    T_s.append(mr.MatrixExp6((s) * theta[i]))
    i = i + 1

Ttheta_s = np.eye(4)
for t in T_s:
    Ttheta_s = Ttheta_s @ t
Ttheta_s = np.round(Ttheta_s @ M, decimals=3)


Ttheta2_s = np.round(mr.FKinSpace(M,Slist,theta),decimals = 3)              # Question 4

print('\n ###################### s-Frame ######################')
print('Using MatrixExp6 function: \n',Ttheta_s)
print('Using FKinSpace function: \n',Ttheta2_s)


# B Frame
Blist = np.array([[0,0,0,0,0,0],[0,1,1,1,0,0],[1,0,0,0,0,1],
                  [0,1+sq3,2+sq3,2,0,0],[1+sq3,0,0,0,0,0],[0,-1-sq3,-1,0,1,0]]) # Question 3

semat_b = []
for b in np.transpose(Blist):
    semat_b.append(mr.VecTose3(b))

T_b = []
i = 0 
for b in semat_b:
    T_b.append(mr.MatrixExp6((b) * theta[i]))
    i = i + 1

Ttheta_b = M
for t in T_b:
    Ttheta_b = Ttheta_b @ t
Ttheta_b = np.round(Ttheta_b , decimals=3)

Ttheta2_b = np.round(mr.FKinBody(M,Blist,theta),decimals = 3)           # Question 5

print('\n ###################### b-Frame ######################')
print('Using MatrixExp6 function: \n',Ttheta_b)
print('Using FKinBody function: \n',Ttheta2_b)

# For both S and B frames:
# Ttheta calculates exp([Si]theta_i) --> Ttheta using MatrixExp6 function
# Ttheta2 directly calculates Ttheta using FKinSpace.
# Ttheta = Ttheta2
