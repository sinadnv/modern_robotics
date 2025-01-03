# Modern Robotics Course 2, Week 3 Final Quiz

import numpy as np
import modern_robotics as mr


########## Question 1 
x0, y0 = 1, 1
x = np.transpose(np.array([x0,y0]))
xd, yd = 3, 2
root = np.transpose(np.array([xd, yd]))

err = 1
counter = 0
while err > 1e-3:
    #print('Iteration #: ', counter,' ; Answer: ', x)
    g = np.transpose(np.array([x[0]**2-9, x[1]**2-4]))
    dg = np.array([[2*x[0], 0], [0, 2*x[1]]])
    xold = x
    x = x - np.linalg.pinv(dg) @ g
    err = np.linalg.norm(x-xold)/np.linalg.norm(xold)
    counter = counter + 1
#print('Iteration #: ', counter,' ; Final Answer: ', x)


########## Question 2 
Tsd = np.array([[-.585, -.811, 0, .076],[.811,-.585,0,2.608],[0,0,1,0],[0,0,0,1]])
M = np.transpose(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[3,0,0,1]]))
eomg, ev = 0.001, 0.0001
thetalist = np.transpose(np.array([np.pi/4, np.pi/4, np.pi/4]))

# Method 1: Use inverse kinematics in space configuration
Slist = np.transpose(np.array([[0,0,1,0,0,0],[0,0,1,0,-1,0],[0,0,1,0,-2,0]]))
theta_space = mr.IKinSpace(Slist, M, Tsd, thetalist, eomg, ev)
ans2a = theta_space[0]

# Method 2: Use inverse kinematics in body configuration
Blist = np.transpose(np.array([[0,0,1,0,3,0],[0,0,1,0,2,0],[0,0,1,0,1,0]]))
theta_body = mr.IKinBody(Blist, M, Tsd, thetalist,eomg,ev)
ans2b = theta_body[0]

# Method 3: Implement Newton-Raphson iterative method without using IK functions
len_wb, len_vb = 1, 1
while len_wb > eomg or len_vb > ev:
    Tsb = mr.FKinBody(M, Blist, thetalist)
    Jb = mr.JacobianBody(Blist, thetalist)
    Jbinv = np.linalg.pinv(Jb)
    twist_b = mr.se3ToVec(mr.MatrixLog6(np.linalg.inv(Tsb) @ Tsd))
    thetalist = thetalist + Jbinv @ twist_b
    w_b = twist_b[0:3]
    v_b = twist_b[3:6]
    len_wb = np.linalg.norm(w_b)
    len_vb = np.linalg.norm(v_b)
theta_nr = thetalist
ans2c = theta_nr


# if (np.round(ans2a,2) == np.round(ans2b,2)).all and (np.round(ans2a,2) == np.round(ans2c,2)).all:
#     print('All Three Calculated values for theta_d are similar and equal to:  ', np.round(ans2a,2))


########## Project

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return void: Does not return any values. It only prints the designated values.

    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        thetaMat = np.array([[1.5, 2.5, 3.],
                            [1.58239319, 2.97475147, 3.15307873]
                            [1.57073819, 2.999667,   3.14153913]])

        Priniting the following values per iteration:
            Joint effector
            SE(3) end−effector config
            error twist V_b
            angular error magnitude ∣∣omega_b∣∣
            linear error magnitude ∣∣v_b∣∣
        Also prints a matrix of all joint values per each iteration (thetaMat). The matrix is saved to a csv file.

                   
    """

    thetalist = np.array(thetalist0).copy()
    thetaMat = np.array(thetalist).copy()           # thetaMat collection of all thetas for each iteration, first row is thetalist0
    ### The following portion of the code is from the original MR function, IKinBody, and does not need additional commenting
    i = 0
    maxiterations = 20
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, \
                                                         thetalist)), Vb)
        i = i + 1
        Vb \
        = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, \
                                                       thetalist)), T)))
        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
        
        ### End of original MR IKinBody code
        
        # Printing variables requested in the exercise
        print('Iteration ', i, ':')
        print('Joint effector', thetalist)
        print('SE(3) end−effector config: ', mr.FKinBody(M, Blist, thetalist))
        print('error twist V_b: ', Vb)
        print('angular error magnitude ∣∣omega_b∣∣: ', np.linalg.norm([Vb[0], Vb[1], Vb[2]]))
        print('linear error magnitude ∣∣v_b∣∣: ', np.linalg.norm([Vb[3], Vb[4], Vb[5]]) )
        # adding updated thetalist to the end of thetaMat (as a new row)
        thetaMat = np.vstack([thetaMat, thetalist])

    np.savetxt("iterates.csv", thetaMat, delimiter=",", fmt="%.5f")
    return thetaMat

l1, l2 = .425, .392,  
w1, w2 = .109, .082
h1, h2 = .089, .095
M = np.transpose(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[l1+l2, w1+w2, h1-h2, 1]]))
Blist = np.transpose(np.array([[0,1,0,w1+w2,0,l1+l2],[0,0,1, h2, -l1-l2,0],[0,0,1,h2,-l2,0],[0,0,1,h2,0,0],[0,-1,0,-w2,0,0],[0,0,1,0,0,0]]))
Tsd = np.array([[0,1,0,-.5],[0,0,-1,.1],[-1,0,0,.1],[0,0,0,1]])
eomg, ev = .001, .0001
thetalist0 = np.array([np.pi, 0, np.pi/2,np.pi/2,-3*np.pi/2,np.pi/2])

IKinBodyIterates(Blist, M, Tsd, thetalist0, eomg, ev)
