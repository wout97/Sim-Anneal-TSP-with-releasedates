'''
About this program
------------------------------

This program uses a metaheuristic called "Simulated Annealing" to solve
the travelling salesman algorithm with release dates as described in "Archetti et. all(2017)[1]".

When finisched executing a few graphs will show the current solution, 
how the temperature/propability evolved and how the best solution changed through the process.

As described in the paper[1] we will use the fact that there exists an optimal solution with a waiting time only when starting the first route.
The following routes are expected to start immideately. 

A solution will be represented by list of lists. 
The lists in the list represent different routes, and thus contains nodenumbers corresponding to the problemNodes.
Every route is assumed to start and end in the depotNode, but is not explicity in the list.

Given a possible solution the execution time can be deterministicly determined, by calculating the  smallest witing time that results
into a feasable solution, added by the total execution time of all routes.


Reference:
-----------------------------------------------------------
[1]Archetti, C., Feillet, D., Mor, A. & Speranza, M.G. (2018). An iterated local search for the Traveling Salesman Problem with release dates and completion time
minimization. Computers & Operations Research, 98: 24-37.

Example:
---------------------------------------------------------

A possible solution containing three routes.
[
 [ Node1, Node2, Node3],
 [ Node4, Node5, Node6],
 [ Node7, Node8, node9],
]


Permutations 

1) Merge Two Routes:

[                                             [   
 [ Node1, Node2, Node3],     ==>               [ Node1, Node2, Node3, Node4, Node5, Node6]
 [ Node4, Node5, Node6]                       ]
]

2) Split One Route:

[                                               [   
 [Node1, Node2, Node3]     ==>                   [ Node1, Node2],                       
]                                                [ Node 3]
                                                ]

3) Swap Two Routes:

[                                              [   
 [ Node1, Node2, Node3],     ==>                [ Node4, Node5, Node6], 
 [ Node4, Node5, Node6]                         [ Node1, Node2, Node3]
]                                              ] 

4) Swap Nodes:

[                                               [   
 [ Node1, Node2, Node3],     ==>                 [ Node1, Node2],  
 [ Node4, Node5, Node6]                          [ Node4, Node5, Node6, Node3]
]                                               ] 

'''
##############
'''IMPORTS'''
##############

import random
import math
from time import sleep
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt # to plot
import matplotlib as mpl
from mpl_toolkits import mplot3d
from colour import Color
import copy #copy np array
import scipy.constants as sc
from scipy import optimize  # to compare

###############################
'''VISUALIZATION FUNCTIONS'''
###############################

def visualise_solution_2d(depotNode, solution, nodes):
    #create color gardient
    red = Color("green")
    colors = list(red.range_to(Color("red"),len(solution)))
    CoDepot = nodes
    ax = plt.axes()
    #Previous coordinates (init at same as first)
    Co1P = nodes[:,0][solution[0]]
    Co2P = nodes[:,1][solution[0]]
    CoDepotX = nodes[:,0][depotNode]
    CoDepotY = nodes[:,1][depotNode]
    iterator =0
    totalCost = 0
    for route in solution:
        Co1P = CoDepotX
        Co2P = CoDepotY
        for node in route:
            CoX = nodes[:,0][node]
            CoY = nodes[:,1][node]
            CoZ = nodes[:,2][node]
            #print("node " + str(x) + " from " + str(Co1P) +" "+ str(Co2P) + "to " + str(Co2)+ " " + str(Co1))
            plt.scatter(CoX,CoY,color=colors[iterator].rgb,marker="o")
            #draw line from previous point to new point
            ax.arrow(Co1P, Co2P  , CoX-Co1P, CoY-Co2P, color=colors[iterator].rgb)
            cost = int(round(np.sqrt((CoX-Co1P)**2 + (CoY-Co2P)**2)))
            #cost = nodes[:,2][x]
            ax.annotate(str(cost) + " " +str(CoZ), (CoX,CoY),fontsize=10)
            totalCost += cost 
            if iterator == 0  :
                cost = "D"
            elif iterator == len(solution) - 1:
                cost=""
            Co1P = CoX
            Co2P = CoY
        CoX = CoDepotX
        CoY = CoDepotY
        ax.arrow(Co1P, Co2P , CoX-Co1P, CoY-Co2P, color=colors[iterator].rgb)
        iterator +=1
    plt.gcf().text(0, 0, "Total cost = " + str(totalCost), fontsize=14)
    plt.title('Solution in 2D',fontsize=10)
    plt.show()
#Not used anymore
def visualise_solution_3d(solution, nodes):
    '''
    plot all points
    '''
    red = Color("green")
    colors = list(red.range_to(Color("red"), len(solution)))
    global plt
    ax = plt.axes(projection='3d')
    i = 0
    x2 = nodes[:,0][solution[0]]
    y2 = nodes[:,1][solution[0]]
    z2 = nodes[:,2][solution[0]] 
    for x in solution:
        x1 = nodes[:,0][x]
        y1 = nodes[:,1][x]
        z1 = nodes[:,2][x]
        ax.scatter3D(x1, y1, z1, color=colors[i].rgb)
        ax.plot([x1,x2],[y1,y2],[z1,z2],color = colors[i].rgb)
        x2 = x1
        y2 = y1
        z2 = z1
        i+=1
    plt.show()
def visualise_problem_3d(nodes, cost):
    '''
    plot all points after #'cost' seconds
    '''
    global plt
    ax = plt.axes(projection='3d')
    ax.scatter3D(nodes[:,0], nodes[:,1], nodes[:,2]-cost)
    plt.show()

###############################
'''GENERATE  PROBLEM'''
###############################

#generate a random problem
def nodeGenerator(width, height, maxWait, nodesNumber):
    '''
    create #'nodesnumner'-random nodes on space [width,height,maxWait] 
    '''
    xs = np.random.randint(width, size=nodesNumber)
    ys = np.random.randint(height, size=nodesNumber)
    zs = np.random.randint(maxWait,size=nodesNumber)
    return np.column_stack((xs, ys,zs))
#strip a problem from DAT file
def stripDatFiles(fileLocation):
    datContent = [i.strip().split() for i in open(fileLocation).readlines()]
    dataContentCoords = datContent[5:]
    dataContentCoordsWithoutZ = []
    for l in dataContentCoords:
        dataContentCoordsWithoutZ.append([int(l[0]), int(l[1]), int(l[6])])
    return dataContentCoordsWithoutZ

###############################
'''GENERATE INITIAL SOLUTION'''
###############################

def InitialSolution(nodes):
    '''
    Computes the initial solution with max waitingTime
    '''
    listNodes = nodes.tolist()
    #get a random Node out of all nodes as DEPOT node
    #depot = random.randrange(len(nodes))
    depot = 0
    #generate a list of nodes that are not in path yet
    nodes_to_visit = list(range(len(nodes)))
    nodes_to_visit.remove(depot)  #remove depot one
  
    
    #initial values
    totalCost = 0
    timeElapsed = 0
    startWaitingTime = 0
    result =[]
    nearestNode = depot

    #iterate while not empty
    while nodes_to_visit:
        selectNodes = []
        biggestWaitingTime = 0
        smallestCost = 1000000000000000

        #check all nodes, to decide what node has biggest waiting time
        for x in nodes_to_visit:
            #calculate cost of next node
            cost =np.sqrt((nodes[nearestNode][0] - nodes[x][0])**2 + (nodes[nearestNode][1] - nodes[x][1])**2)
            waitingTimeCurrent = nodes[x][2]
            if waitingTimeCurrent > biggestWaitingTime:
                    biggestWaitingTime = waitingTimeCurrent
            if cost < smallestCost:
                smallestCost = cost
                nextNode = x
        nearestNode = nextNode
        totalCost += smallestCost
        #indexNearest = listNodes.index(nearestNode.tolist())
        nodes_to_visit.remove(nextNode)
        result.append(nextNode)
    print(result)
    result = [result]
    print(result)
    return result, biggestWaitingTime, depot

####################################
'''GENERATE NEIGBOURING SOLUTIONS'''
####################################
#Init chances of certain permutations
chanceChange = 1
chanceMerge = 0
chanceSplit = 0
chanceSwap = 0

indicator = 0
#generate neigbour
def anneal( temp, problemNodes ):
    '''
    Annealing
    '''
    global currSolution
    global bestSolution
    global bestCost
    global currCost
    global chanceMerge
    global chanceSplit
    global chanceSwap
    global chanceChange
    candidate = copy.deepcopy(currSolution)

    if random.randint(0,1) < chanceMerge:
        #print("Merge")
        #print(len(candidate))
        #print(candidate)
        if (len(candidate) == 1):
            r1 = 0
            r2 = 0
        else:
            r1 = random.randint(0, len(candidate)-1)
            r2 = random.randint(0, len(candidate)-1)
            if r1 != r2:
                candidate = copy.deepcopy(mergeTwoRoutes(r1,r2, candidate))
        #print(candidate)
    if random.randint(0,1) < chanceSplit:
        #print("Split")
        #print(len(candidate))
        #print(candidate)
        if (len(candidate) == 1):
            r1 = 0
        else:
            r1 = random.randint(0, len(candidate)-1)
        if (len(candidate[r1]) == 1):
            r1 = 0
        else:
            indexToSplit = random.randint(0, len(candidate[r1])-1)
            candidate = copy.deepcopy(splitOneRoute(r1,indexToSplit, candidate))
        #print(candidate)
    if random.randint(0,1) < chanceSwap:
        #print("Swap")
        #print(len(candidate))
        #print(candidate)
        if (len(candidate) == 1):
            r1 = 0
            r2 = 0
        else:
            r1 = random.randint(0, len(candidate)-1)
            r2 = random.randint(0, len(candidate)-1)
            candidate = copy.deepcopy(swapTwoRoutes(r1,r2, candidate))
        #print(candidate)
    if random.randint(0,1) < chanceChange:
        #print("change")
        #print(len(candidate))
        #print(candidate)
        if (len(candidate) == 1):
            r1 = 0
            r2 = 0
        else:
            r1 = random.randint(0, len(candidate)-1)
            r2 = random.randint(0, len(candidate)-1)

        #if (len(candidate[r1]) == 1):
            #r1 =0
        
        indexR1 = random.randint(0, len(candidate[r1])-1)
        indexR1B = random.randint(0, len(candidate[r2])-1)

        #r2 = random.randint(0, len(candidate)-1)
        indexR2 = random.randint(0, len(candidate[r1])-1)
        indexR2B = random.randint(0, len(candidate[r2])-1)
        candidate = copy.deepcopy(changeNodeLocation(r1,indexR1, r1, indexR2, candidate))
        candidate = copy.deepcopy(changeNodeLocation(r2,indexR1B, r2, indexR2B, candidate))

        #print(candidate)
    return candidate
#Helperfunctions with the four permutations
def mergeTwoRoutes(r1, r2, array ):
    array[r1] += array[r2]
    array.pop(r2)
    return array
def splitOneRoute(r1, indexToSplit, array ):
    array.insert(r1+1, array[r1][indexToSplit:])
    array[r1] = array[r1][:indexToSplit]
    if array[r1] == []:
        array.pop(r1)
    return array
def swapTwoRoutes(r1, r2, array):
    array[r1], array[r2] = array[r2], array[r1]
    return array
def changeNodeLocation(r1,indexR1, r2, indexR2, array):
    #array[r2].insert(indexR2, array[r1].pop(indexR1))
    array[r1][indexR1], array[r1][indexR2] = array[r1][indexR2], array[r1][indexR1]
    if array[r1] == []:
        array.pop(r1)
    return array
def changeChances():
    global chanceChange
    global chanceMerge
    global chanceSplit
    global chanceSwap
    global indicator
    
    if indicator == 0:
        
        chanceChange = 1
        chanceMerge = 0
        chanceSplit = 0
        chanceSwap = 0

    elif indicator ==1:
        chanceChange = 1
        chanceMerge = 1
        chanceSplit = 0
        chanceSwap = 1
    elif indicator ==2:
        chanceChange = 1
        chanceMerge = 0
        chanceSplit = 1
        chanceSwap = 0
    elif indicator ==3:
        chanceChange = 1
        chanceMerge = 0
        chanceSplit = 0
        chanceSwap = 1
    indicator += 1 
    if indicator == 4:
        indicator =0

####################################
'''ACCEPT/REJECT BASED ON COST/TEMP'''
####################################
#calculate acceptance prob with temparature
def acceptance_probability(candidateCost, currCost, temp):
    '''
    Acceptance probability as described in:  
    '''
    #return math.exp(-abs(candidateCost - currCost) / temp)
    #return np.exp( abs(candidateCost - currCost) / (temp*sc.k))
    #return np.exp( -abs(candidateCost - currCost) / (temp*sc.k))
    return np.exp( - abs(candidateCost - currCost) / (temp))
#calculate waiting time for candidate solution
def cost(depotNode, candidateSolution, nodes):
    totalCost = 0
    iterator = 0
    listNodes = copy.deepcopy(nodes.tolist())
    timeElapsed = 0
    initalWaitingTime = 0
    timeSinceLastLeft = 0
    routeCost = 0

    for route in candidateSolution:
        prevNode = depotNode
        routeCost = 0
        for newNode in route:
           #calculate square distance
            cost = math.sqrt( (listNodes[prevNode][0] - listNodes[newNode][0])**2 + (listNodes[prevNode][1] - listNodes[newNode][1])**2 )
            routeCost += cost
            if iterator == 0 and listNodes[newNode][2] > initalWaitingTime:
                initalWaitingTime = listNodes[newNode][2] 
            else:
                if timeSinceLastLeft + initalWaitingTime <  listNodes[newNode][2]:
                    initalWaitingTime += (listNodes[newNode][2] - (timeSinceLastLeft + initalWaitingTime))
            prevNode = newNode
            totalCost += cost
            timeElapsed += cost
        cost = math.sqrt( (listNodes[prevNode][0] - listNodes[depotNode][0])**2 + (listNodes[prevNode][1] - listNodes[depotNode][1])**2 )
        totalCost += cost
        timeElapsed += cost
        prevNode = depotNode
        routeCost += cost
        timeSinceLastLeft +=routeCost 
        iterator +=1
    return totalCost, initalWaitingTime

#Accept or rejects candidate solution
def accept(candidate, candidateCost, temp):
    '''
    Accept with probability 1 if candidate solution is better than
    current solution, else accept with probability
    '''
    global currSolution
    global bestSolution
    global bestCost
    global currCost
    if candidateCost <= currCost:
        currCost = copy.deepcopy(candidateCost)
        currSolution = candidate
        if candidateCost < bestCost:
            print("best found---------------------------------")
            bestCost = candidateCost
            bestSolution = copy.deepcopy(candidate)
    else:
        if random.random() < acceptance_probability(candidateCost, currCost, temp):
            currCost = candidateCost
            currSolution = copy.deepcopy(candidate)


#Get a Problem

#problemNodes = nodeGenerator(500,500,1000, 10)
#problemNodes = np.array([[40,	50,	 0],[45,	68,0 ],[45,	70,	6],[42,	66,	 7],[42,	68,	 7],[42,	65,	6],[40,	69,	20],[40,	66,	15],[38,	68,	20],[38,	70,	 23],[35,	66,	 10]])
#problemNodes = np.array([[40,	50,	  0],[45,	68,	 	0],[45,	70,		27],[42,	66,	 	35],[42,	68,	 	34],[42,	65,	 	26],[40,	69,	 	89],[40,	66,	 	66],[38,	68,	 	88],[38	,70,	 	102],[35	,66	, 	46],[35	,69	 ,	20],[25	,85,	106],[22	,75,	7],[22	,85	 ,	64],[20	,80	,	81],[20	,85	 ,9],[18	,75	 ,	90],[15	,75	 ,60],[15	,80	, 32],[30	,50,	74]])
problemNodes = np.array(stripDatFiles("C101_0.5.dat"))

# create initial solution
seedSolution, seedSolutionCost, depotNode = InitialSolution(problemNodes)
seedSolutionCostRoute, waitingTime = cost(depotNode, seedSolution, problemNodes)
print(waitingTime)
print(seedSolutionCostRoute)
seedSolutionCost = seedSolutionCostRoute + waitingTime
print(waitingTime)

#visualise solution
visualise_solution_2d(depotNode,seedSolution, problemNodes)


#start Annealing

#init values
currSolution = copy.deepcopy(seedSolution)
currCost = seedSolutionCost
bestSolution = copy.deepcopy(seedSolution)
bestCost = seedSolutionCost
sampleSize = len(problemNodes)
temp = 100000
tempDecrease = 1
stoppingTemp = 0
iteration = 0
stoppingIteration = 200000

#debug info
print("start annealing")
print("seedsolution = " + str(seedSolution))
print("seedsolutioncost = " + str(seedSolutionCost))
print("bestCost = " + str(bestCost))
print("bestSolution = " + str(bestSolution))

#variables for graphs at the end
costArray = []
candidateCost = 0
tempArray =[]
bestCostArray=[]
probArray =[]

while iteration < stoppingIteration and temp > 0:
    #select new candidate and calculate its cost
    candidate = copy.deepcopy(anneal( temp, problemNodes))
    initWaitingTime, candidateCostRoute = cost(depotNode, candidate, problemNodes)
    candidateCost = initWaitingTime + candidateCostRoute
    #accept/ deny new candidate with prob-function
    accept(candidate, candidateCost, temp)
    #update temperature and iterator

    iteration += 1
    changeChances()

    #Save some information for graph at end
    if iteration % 10 == 0:
        costArray.append(currCost)
        tempArray.append(temp)
        bestCostArray.append(bestCost)
        aP = acceptance_probability(candidateCost, currCost, temp)
        if aP ==1:
            pass
        else:
            probArray.append( aP)

    if iteration % 1000 == 0:
        print("Iteration : " + str(iteration) + " | bestCost = " + str(bestCost) + " | currCost=" + str(currCost) + "| temp=" + str(temp) + " | prob="+ str(acceptance_probability(candidateCost, currCost, temp) ))
        print("candidateCost = " + str(candidateCost))
        print(currSolution)
        print(bestSolution)
    if (temp > tempDecrease):
        temp -= tempDecrease
    else:
        temp = 0.00001
    

#display results
print('Minimum weight: ', bestCost)
print(cost(depotNode, currSolution, problemNodes))
visualise_solution_2d(depotNode, bestSolution, problemNodes)
plt.plot(costArray)
plt.show()
plt.plot(bestCostArray)
plt.show()
plt.plot(tempArray)
plt.show()
plt.plot(probArray)
plt.show()