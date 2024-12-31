# Modern Robotics Course 2, Week 2 Final Quiz

import numpy as np
import modern_robotics as mr

########## Question 1
thetalist = np.transpose(np.array([0, np.pi/4,0,]))
#Slist = np.transpose(np.array([[1,0,0],[1,0,-1],[1,1/np.sqrt(2),1+1/np.sqrt(2)]]))
# JacbianSpace function supports 6DoF Slist, so added wx, wz, vz (all equal to zero) to the matrix as well.
Slist = np.transpose(np.array([[0,0,1,0,0,0],[0,0,1,0,-1,0],[0,0,1,0,-2,0]]))

Js = mr.JacobianSpace(Slist,thetalist)
#Remove non-zero rows to go back to 3DOF.
Js = Js[~np.all(Js == 0, axis =1)]

Fs = np.transpose(np.array([0,2,0]))
tau = np.transpose(Js) @ Fs
ans1 = tau


########## Question 2
l1, l2, l3, l4 = 1, 1, 1, 1
theta1, theta2, theta3, theta4 = 0, 0, np.pi/2, -np.pi/2
thetalist = np.transpose(np.array([theta1, theta2, theta3, theta4]))
# Removed wx, wy, vz from Fb so that its dimensions match Jb 
Fb = np.transpose(np.array([10,10,10]))

Jb = np.array(np.transpose([[1, l3 * np.sin(theta4) + l2 * np.sin(theta3+theta4) + l1 * np.sin(theta2+theta3+theta4), l4 + l3 * np.cos(theta4) + l2 * np.cos(theta3+theta4) + l1 * np.cos(theta2+theta3+theta4)],
                           [1, l3 * np.sin(theta4) + l2 * np.sin(theta3+theta4), l4 + l3 * np.cos(theta4) + l2 * np.cos(theta3+theta4)],
                           [1, l3 * np.sin(theta4), l4 + l3 * np.cos(theta4)],
                           [1, 0, l4]]))

tau = np.transpose(Jb) @ Fb
ans2 = tau
print(ans2)

########## Question 3
Slist = np.transpose(np.array([[0,0,1,0,0,0],[1,0,0,0,2,0],[0,0,0,0,1,0]]))
thetalist = np.transpose(np.array([np.pi/2, np.pi/2, 1]))

# Method 1: Calculte Js manually
Js1 = np.zeros((6, len(thetalist)))
T = np.eye(4)
for i in range(1, len(thetalist)):
    T = T @ mr.MatrixExp6(mr.VecTose3(np.array(Slist)[:, i - 1] * thetalist[i - 1]))
    Js1[:, i] = mr.Adjoint(T) @ np.array(Slist)[:, i]

# Method 2: Calculte Js directly using JacobianSpace Function
Js2 = mr.JacobianSpace(Slist,thetalist)

Js1 == Js2


########## Question 4
Blist = np.transpose(np.array([[0,1,0,3,0,0],[-1,0,0,0,3,0],[0,0,0,0,0,1]]))
thetalist = np.transpose(np.array([np.pi/2, np.pi/2, 1]))

# Method 1: Calculte Jb manually
Jb1 = np.zeros((6, len(thetalist)))
T = np.eye(4)
for i in range(len(thetalist)-2, -1, -1):
    T = T @ mr.MatrixExp6(mr.VecTose3(np.array(Blist)[:, i + 1] * -thetalist[i + 1]))
    Jb1[:, i] = mr.Adjoint(T) @ np.array(Blist)[:, i]

# Method 2: Calculte Js directly using JacobianBody Function
Jb2 = mr.JacobianBody(Blist,thetalist)


########## Question 5 and 6
Jb = np.array([[0,-1,0,0,-1,0,0],
               [0,0,1,0,0,1,0],
               [1,0,0,1,0,0,1],
               [-.105,0,.006,-.045,0,.006,0],
               [-.889,.006,0,-.844,.006,0,0],
               [0,-.105,.889,0,0,0,0]])
Jv = Jb[3:6,:]

A = Jv @ np.transpose(Jv)

[eValue, eVector] = np.linalg.eig(A)
sortedIndices = np.argsort(eValue)
eValueSorted = eValue[sortedIndices]
eVectorSorted = eVector[:, sortedIndices]

ans5 = eVectorSorted[:,-1]
ans6 = np.sqrt(eValueSorted[-1])
