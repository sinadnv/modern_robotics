# This code provides Jacobian newton-raphson method, a solution to example 6.1 in the MR book.

import numpy as np
import modern_robotics as mr


Tsd = np.array([[-.5, -.866, 0, .366],[.866, -.5, 0, 1.366]
                ,[0, 0, 1, 0], [0, 0, 0, 1]])
Blist = np.array(np.transpose([[0,0,1,0,2,0],[0,0,1,0,1,0]]))
theta_d = np.array(np.transpose([np.pi/6, np.pi/2]))
thetalist = np.array([0, np.pi/6])
M = np.array(np.transpose([[1,0,0,0],[0,1,0,0],[0,0,1,0],[2,0,0,1]]))

len_wb, len_vb = 1, 1
err_w, err_v = .001, .0001
while len_wb > err_w:
    Tsb = mr.FKinBody(M, Blist, thetalist)
    Jb = mr.JacobianBody(Blist, thetalist)
    Jbinv = np.linalg.pinv(Jb)
    twist_b = mr.se3ToVec(mr.MatrixLog6(np.linalg.inv(Tsb) @ Tsd))
    thetalist = thetalist + Jbinv @ twist_b
    w_b = twist_b[0:3]
    v_b = twist_b[3:6]
    len_wb = np.linalg.norm(w_b)
    len_vb = np.linalg.norm(v_b)

print(np.round(np.rad2deg(thetalist),2))
