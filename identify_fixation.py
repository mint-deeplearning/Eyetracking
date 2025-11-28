
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ivt(data,v_threshold, sample_gap, verbose=0):

    Xs = data[:,0]
    Ys = data[:,1]

    diffX = [] 
    diffY = [] 

    for i in range(len(data) - 1):
        diffX.append(float(Xs[i+1]) - float(Xs[i]))
        diffY.append(float(Ys[i+1]) - float(Ys[i]))
        
    Velocity = []

    for i in range(len(diffX)):
        Velocity.append(diffX[i] + diffY[i])
        

    velocity=np.divide(Velocity, sample_gap)        # 2
    velocity=np.absolute(velocity)

    # global mvmts
    mvmts = []
    
    for v in velocity:
        if (v < v_threshold):
            mvmts.append(1)
        else:
            mvmts.append(2)
    if verbose==1: ## to show charts
        print(('minimum_velocity=',min(velocity)))
        print(('maximum_velocity=',max(velocity)))
        plt.plot(velocity)
        plt.xlabel("Time [ms]")
        plt.ylabel("Velocity [px/ms]")
        plt.show()

    return mvmts,velocity

def ivt_1(data,v_threshold, sample_gap, verbose=0):

    Xs = data[:,0]
    Ys = data[:,1]

    diffX = [] 
    diffY = [] 

    for i in range(len(data) - 1):
        diffX.append(float(Xs[i+1]) - float(Xs[i]))
        diffY.append(float(Ys[i+1]) - float(Ys[i]))

    distance = np.sqrt(np.power(diffX,2) + np.power(diffY,2))
    velocity = np.divide(distance, sample_gap) #
    

    # global mvmts
    mvmts = []

    for v in velocity:
        if (v < v_threshold):
            mvmts.append(1)
        else:
            mvmts.append(2)
    if verbose==1: ## to show charts
        print(('minimum_velocity=', min(velocity)))
        print(('maximum_velocity=', max(velocity)))
        plt.plot(velocity)
        plt.xlabel("Time [ms]")
        plt.ylabel("Velocity [px/ms]")
        plt.show()

    return mvmts,velocity

def ivt_2(data,v_threshold, verbose=0):
    # sample_gap 采样间隔  毫秒
    
    Xs = data[:, 3]
    Ys = data[:, 4]
    Ts = data[:, 5]
    diffX = [] 
    diffY = [] 
    sample_gap = []
    for i in range(len(data) - 1):
        diffX.append(float(Xs[i+1]) - float(Xs[i]))
        diffY.append(float(Ys[i+1]) - float(Ys[i]))
        sample_gap.append((float(Ts[i+1]) - float(Ts[i]))*1000)

    distance = np.sqrt(np.power(diffX,2) + np.power(diffY,2))
    velocity = np.divide(distance, sample_gap) # 2ms gap!
    # velocity = np.absolute(velocity)
    # print('distance: ', distance.shape)
    # print('sample gap: ', len(sample_gap))
    # print('velocity: ', velocity.shape)
    # global mvmts
    mvmts = []
    
    for v in velocity:
        if (v < v_threshold):
            mvmts.append(1)
        else:
            mvmts.append(2)
    if verbose==1: ## to show charts
        print(('minimum_velocity=', min(velocity)))
        print(('maximum_velocity=', max(velocity)))
        plt.plot(velocity)
        plt.xlabel("Time [ms]")
        plt.ylabel("Velocity [px/ms]")
        plt.show()

    return mvmts,velocity