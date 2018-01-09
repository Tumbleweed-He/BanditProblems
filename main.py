# -*- coding: utf-8 -*-
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def test_norm ():
    mu = 0
    sigma = 1
    x = np.arange(-5, 5, 0.1)
    y = stats.norm.pdf(x, 0, 1)
    print(y)
    plt.plot(x, y)
    plt.title('Normal:$\mu$=%.1f, $\sigma^2$=%.1f' % (mu, sigma))
    plt.xlabel('x')
    plt.ylabel('Probability density', fontsize = 15)
    plt.show()
#print(stats.norm.rvs(0, 1, 10))
#test_norm()

def main():
    totalRound = 1000;
    i = 0
    while i < 10:
        i += 1
        means = stats.norm.rvs(0, 1, 10)
        res1 = run(0.1, totalRound, means)
        res2 = run(0.01, totalRound, means)
        res3 = run(0, totalRound, means)        
        x = np.arange(1, totalRound + 1, 1)
        plt.plot(x, res1['optimalPercent'], label = "$e = 0.1$", color = "black", linewidth = 1)
        plt.plot(x, res2['optimalPercent'], label = "$e = 0.01$", color = "red", linewidth = 1)
        plt.plot(x, res3['optimalPercent'], label = "$e = 0$", color = "green", linewidth = 1)
        plt.title('Bandit Problems')
        plt.xlabel('steps')
        plt.ylabel('Optimal Action Percent(%)', fontsize = 15)
        plt.show()
    
    '''
        plt.plot(x, res1['avgRew'], label = "$e = 0.1$", color = "black", linewidth = 1)
        plt.plot(x, res2['avgRew'], label = "$e = 0.01$", color = "red", linewidth = 1)
        plt.plot(x, res3['avgRew'], label = "$e = 0$", color = "green", linewidth = 1)
        plt.title('Bandit Problems')
        plt.xlabel('steps')
        plt.ylabel('Average Reward', fontsize = 15)
        plt.show()
    '''
    
def run(explorationRate, totalRound, means):
    totalReward = 0
    currentRound = 1;
    actionNums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    actionValues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    greedyAction = np.argmax(means)
    y = []
    greedyActionPercent = []
    greedyActionTimes = 0
    while currentRound <= totalRound:
        realExplorationRate = random.random()
        if realExplorationRate >= explorationRate: #exploitation
            actions = foundMaxValueIndices(actionValues)
            index = int(random.random() * len(actions))
            takeAction = actions[index]
            reward = stats.norm.rvs(means[takeAction], 1)
            actionNums[takeAction] = actionNums[takeAction] + 1
            actionValues[takeAction] = actionValues[takeAction] + (reward - actionValues[takeAction]) / actionNums[takeAction]
            totalReward += reward
            if takeAction == greedyAction:
                greedyActionTimes += 1
            greedyActionPercent.append(greedyActionTimes / currentRound)
        else: #exploration
            actions = foundMaxValueIndices(actionValues)
            maxValueAction = actions[int(random.random() * len(actions))]
            notMaxValueActions = []
            for i in range(10):
                if i != maxValueAction:
                    notMaxValueActions.append(i)
            takeAction = notMaxValueActions[int(random.random() * len(notMaxValueActions))]
            reward = stats.norm.rvs(means[takeAction], 1)
            actionNums[takeAction] = actionNums[takeAction] + 1
            actionValues[takeAction] = actionValues[takeAction] + (reward - actionValues[takeAction]) / actionNums[takeAction]
            totalReward += reward
            if takeAction == greedyAction:
                greedyActionTimes += 1
            greedyActionPercent.append(greedyActionTimes / currentRound)
        y.append(totalReward / currentRound)
        currentRound += 1
    return {'avgRew':y, 'optimalPercent':greedyActionPercent}
def test():
    t = np.arange(-1, 2, 0.01)
    s = np.sin(2 * np.pi * t)
    plt.plot(t, s)
    plt.show()
    plt.close()
    l = plt.axhline(linewidth=4, color='r')
    plt.axis([-1, 2, -1, 2])
    plt.show()
        
def foundMaxValueIndices(array):
    indices = []
    maxNum = np.max(array)
    for i in range(10):
        if array[i] == maxNum:
            indices.append(i)
    return indices
    

    
if __name__ == "__main__":
    main()
    
