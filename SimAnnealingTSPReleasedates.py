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
import matplotlib.animation as animation

###############################
'''VISUALIZATION FUNCTIONS'''
###############################

def visualise_solution_2d(depotNode, solution, nodes):
    #create color gardient
    red = Color("red")
    colors = list(red.range_to(Color("blue"),len(solution)))
    fig, ax = plt.subplots()
    #init previous coordinates
    #Co1P = nodes[:,0][solution[0]]
    #Co2P = nodes[:,1][solution[0]]
    #coordinates of depot Node
    CoDepotX = nodes[:,0][depotNode]
    CoDepotY = nodes[:,1][depotNode]
    colorIterator =0
    routeIterator = 1
    nodeIterator = 0
    totalCost = 0
    plt.scatter(CoDepotX,CoDepotY,c="b", marker="X")

    #iterate over each route
    for route in solution:
        #initialize  Previous node as DepotNode
        Co1P = CoDepotX
        Co2P = CoDepotY
        nodeP = route[0]
        print("Starting route : " + str(route))
        nodeIterator = 0
        #iterate over all nodes in a route
        for node in route:
            CoX = nodes[:,0][node]
            CoY = nodes[:,1][node]
            CoZ = nodes[:,2][node]
            #print("node " + str(x) + " from " + str(Co1P) +" "+ str(Co2P) + "to " + str(Co2)+ " " + str(Co1))
            plt.scatter(CoX,CoY,color=colors[colorIterator].rgb,marker="o")
            #draw line from previous point to new point
            ax.arrow(Co1P, Co2P  , CoX-Co1P, CoY-Co2P, color=colors[colorIterator].rgb)
            cost = int(round(np.sqrt((CoX-Co1P)**2 + (CoY-Co2P)**2)))
            #cost = nodes[:,2][x]
            ax.annotate(str(routeIterator) + "." + str(nodeIterator), (CoX, CoY),fontsize=10)
            totalCost += cost 
            print("From node " + str(node) + " at (" +  str(Co1P) + ";" + str(Co2P) + ")with waiting time of" + str(CoZ) +" to node " +str(nodeP) + " at (" +str(CoX) + ";" + str(CoY) + ") has a cost of "+ str(cost) +" [total cost so far = "+str(totalCost) + "]" )
            if colorIterator == 0  :
                cost = "D"
            elif colorIterator == len(solution) - 1:
                cost=""
            Co1P = CoX
            Co2P = CoY
            nodeP = node
            nodeIterator +=1
        CoX = CoDepotX
        CoY = CoDepotY
        ax.arrow(Co1P, Co2P , CoX-Co1P, CoY-Co2P, color=colors[colorIterator].rgb)
        colorIterator +=1
        routeIterator +=1
    plt.gcf().text(0, 0, "Total cost = " + str(totalCost), fontsize=14)
    plt.title('Solution in 2D',fontsize=10)
   
    redDot, = plt.plot([0], [np.sin(0)], 'ro')
    def animate(i):
        redDot.set_data(i, np.sin(i))
        return redDot,
    # create animation using the animate() function
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, 6.28 , 0.1), interval=10, blit=True, repeat=True)
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
def anneal( temp, problemNodes, randomFactor ):
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
    chanceMerge = 0.5
    chanceSplit = 0.5
    chanceSwap = 0.5
    chanceChange = 0.5

    for i in range (0, randomFactor):
        if random.randint(0,1) < chanceMerge:
            if (len(candidate) == 1):
                r1 = 0
                r2 = 0
            else:
                r1 = random.randint(0, len(candidate)-1)
                r2 = random.randint(0, len(candidate)-1)
                if r1 != r2:
                    candidate = copy.deepcopy(mergeTwoRoutes(r1,r2, candidate))
        if random.randint(0,1) < chanceSplit:
            if (len(candidate) == 1):
                r1 = 0
            else:
                r1 = random.randint(0, len(candidate)-1)
            if (len(candidate[r1]) == 1):
                r1 = 0
            else:
                indexToSplit = random.randint(0, len(candidate[r1])-1)
                candidate = copy.deepcopy(splitOneRoute(r1,indexToSplit, candidate))
        if random.randint(0,1) < chanceSwap:
            if (len(candidate) == 1):
                r1 = 0
                r2 = 0
            else:
                r1 = random.randint(0, len(candidate)-1)
                r2 = random.randint(0, len(candidate)-1)
                candidate = copy.deepcopy(swapTwoRoutes(r1,r2, candidate))
    
        if random.randint(0,1) < chanceChange:
            if (len(candidate) == 1):
                route1 = 0
                route2 = 0
            else:
               route1 = random.randint(0, len(candidate)-1)
               route2 = random.randint(0, len(candidate)-1)
            indexR1 = random.randint(0, len(candidate[route1])-1)
            indexR1B = random.randint(0, len(candidate[route2])-1)
            #r2 = random.randint(0, len(candidate)-1)
            indexR2 = random.randint(0, len(candidate[route1])-1)
            indexR2B = random.randint(0, len(candidate[route2])-1)
            candidate = copy.deepcopy(changeNodeLocation(route1,indexR1, route1, indexR2, candidate))
            candidate = copy.deepcopy(changeNodeLocation(route2,indexR1B, route2, indexR2B, candidate))
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
        chanceMerge = 1
        chanceSplit = 0
        chanceSwap = 0
    elif indicator ==1:
        chanceChange = 0
        chanceMerge = 1
        chanceSplit = 1
        chanceSwap = 0
    elif indicator ==2:
        chanceChange = 0
        chanceMerge = 0
        chanceSplit = 1
        chanceSwap = 1
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
    return math.exp(-abs(candidateCost - currCost) / temp)
    #return np.exp( abs(candidateCost - currCost) / (temp*sc.k))
    #return np.exp( -abs(candidateCost - currCost) / (temp*sc.k))
    #return np.exp( - abs(candidateCost - currCost) / (temp))
#calculate waiting time for candidate solution
def acceptance_probability2(iterationGiven):
    '''
    Acceptance probability as described in:  
    '''
    if iterationGiven > 200000:
        return 1
    else:
        return iterationGiven/200000
    #return np.exp( abs(candidateCost - currCost) / (temp*sc.k))
    #return np.exp( -abs(candidateCost - currCost) / (temp*sc.k))
    #return np.exp( - abs(candidateCost - currCost) / (temp))
#calculate waiting time for candidate solution
def cost(depotNode, candidateSolution, nodes):
    totalCost = 0
    iterator = 0
    listNodes = copy.deepcopy(nodes.tolist())
    initalWaitingTime = 0
    timeSinceLastLeft = 0
    routeCost = 0
    #iterate over every route
    for route in candidateSolution:
        prevNode = depotNode
        routeCost = 0
        #iterate over nodes in route
        for newNode in route:
            #calculate square distance of two points in route
            cost = math.sqrt( (listNodes[prevNode][0] - listNodes[newNode][0])**2 + (listNodes[prevNode][1] - listNodes[newNode][1])**2 )
            #add to route cost
            routeCost += cost
            #if the waitingTime at node is bigger than the time the driver has left depot add initial waiting time
            if iterator == 0 and listNodes[newNode][2] > initalWaitingTime:
                initalWaitingTime = listNodes[newNode][2] 
            else:
                if timeSinceLastLeft + initalWaitingTime <  listNodes[newNode][2]:
                    initalWaitingTime += (listNodes[newNode][2] - (timeSinceLastLeft + initalWaitingTime))
            prevNode = newNode
            totalCost += cost
        #add cost of return to depot
        cost = math.sqrt( (listNodes[prevNode][0] - listNodes[depotNode][0])**2 + (listNodes[prevNode][1] - listNodes[depotNode][1])**2 )
        totalCost += cost
        prevNode = depotNode
        routeCost += cost
        timeSinceLastLeft +=routeCost 
        iterator +=1
    return totalCost, initalWaitingTime

#Accept or rejects candidate solution
def accept(candidate, candidateCost, temp, iterationGiven):
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
        randInt = random.random()
        acceptanceProb = acceptance_probability(candidateCost, currCost, temp)
        #acceptanceProb = acceptance_probability2(iterationGiven)
        #print("randInt = " + str(randInt) + " accept Prob = " + str(acceptanceProb))
        if randInt < acceptanceProb:
            currCost = candidateCost
            currSolution = copy.deepcopy(candidate)


#Get a Problem

#problemNodes = nodeGenerator(20,20,200, 20)
#problemNodes = np.array([[40,	50,	 0],[45,	68,0 ],[45,	70,	6],[42,	66,	 7],[42,	68,	 7],[42,	65,	6],[40,	69,	20],[40,	66,	15],[38,	68,	20],[38,	70,	 23],[35,	66,	 10]])
#problemNodes = np.array([[40,	50,	  0],[45,	68,	 	0],[45,	70,		27],[42,	66,	 	35],[42,	68,	 	34],[42,	65,	 	26],[40,	69,	 	89],[40,	66,	 	66],[38,	68,	 	88],[38	,70,	 	102],[35	,66	, 	46],[35	,69	 ,	20],[25	,85,	106],[22	,75,	7],[22	,85	 ,	64],[20	,80	,	81],[20	,85	 ,9],[18	,75	 ,	90],[15	,75	 ,60],[15	,80	, 32],[30	,50,	74]])
#problemNodes = np.array(stripDatFiles("C101_0.5.dat"))
problemNodes = np.array(stripDatFiles("R101_0.5.dat"))
#problemNodes = np.array(stripDatFiles("R101_1.5.dat"))
temp =  25000.0
tempDecrease = 0.1
randomFactor = 30
visuals = False
currSolution = []
bestSolution = []
bestCost = 0
currCost = 0
chanceMerge = 0
chanceSplit = 0
chanceSwap = 0
chanceChange = 0
def main(problemNodes, randomFactor, temp, tempDecrease, visuals):
    global currSolution
    global bestSolution
    global bestCost
    global currCost
    global chanceMerge
    global chanceSplit
    global chanceSwap
    global chanceChange
    # create initial solution
    seedSolution, seedSolutionCost, depotNode = InitialSolution(problemNodes)
    seedSolutionCostRoute, waitingTime = cost(depotNode, seedSolution, problemNodes)
    print(waitingTime)
    print(seedSolutionCostRoute)
    seedSolutionCost = seedSolutionCostRoute + waitingTime
    print(waitingTime)
    #visualise solution
    if visuals:
        visualise_solution_2d(depotNode,seedSolution, problemNodes)
    #start Annealing

    #init values
    currSolution = copy.deepcopy(seedSolution)
    currCost = seedSolutionCost
    bestSolution = copy.deepcopy(seedSolution)
    bestCost = seedSolutionCost
    sampleSize = len(problemNodes)
   
    stoppingTemp = 2.5
    iteration = 0
    stoppingIteration = 500000   

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
    stopCondition = False
    prevCurrCost = 0

    while iteration < stoppingIteration and temp > 0 and not stopCondition:
        #select new candidate and calculate its cost
        candidate = copy.deepcopy(anneal( temp, problemNodes, randomFactor))
        initWaitingTime, candidateCostRoute = cost(depotNode, candidate, problemNodes)
        candidateCost = initWaitingTime + candidateCostRoute
        #accept/ deny new candidate with prob-function
        accept(candidate, candidateCost, temp, iteration)
        #update temperature and iterator

        iteration += 1
        
        #changeChances()

        #Save some information for graph at end
        if iteration % 10 == 0:
            costArray.append(currCost)
            tempArray.append(temp)
            bestCostArray.append(bestCost)
            aP = acceptance_probability(candidateCost, currCost, temp)
            #aP = acceptance_probability2(iteration)
            probArray.append(aP)

        if iteration % 10000 == 0:
            print("Iteration : " + str(iteration) + " | bestCost = " + str(bestCost) + " | currCost=" + str(currCost) + "| temp=" + str(temp) + " | prob="+ str(acceptance_probability(candidateCost, currCost, temp) ))
            print("candidateCost = " + str(candidateCost))
            print(currSolution)
            print(bestSolution)
            #visualise_solution_2d(depotNode,currSolution, problemNodes)
            #plt.plot(costArray)
            #plt.draw()
            #plt.pause(0.0001)
        #decrease randomfactor over time
        if iteration % 5000 == 0:
            if randomFactor > 1:
                randomFactor -=1
        #stop if after 5000 iteration no better route has been found!
        if iteration % 50000 == 0:
            if prevCurrCost == currCost:
                stopCondition = True
            prevCurrCost = currCost
        if (temp > tempDecrease):
            temp *= (1 - tempDecrease)
        else:
            temp = 0.00001
    

    #display results
    print('Minimum weight: ', bestCost)
    print(cost(depotNode, currSolution, problemNodes))
    if visuals:
        #visualise_solution_2d(depotNode, bestSolution, problemNodes)
        visualise_solution_2d(depotNode, currSolution, problemNodes)
        plt.plot(costArray)
        plt.title('All current costs',fontsize=10)
        plt.show()
        plt.plot(bestCostArray)
        plt.title('All best costs',fontsize=10)
        plt.show()
        plt.plot(tempArray)
        plt.title('temperature',fontsize=10)
        plt.show()
        plt.plot(probArray)
        plt.title('prob array',fontsize=10)
        plt.show()
    return currCost
def testOne():
    temp =  25000.0
    tempDecrease = 0.1
    visuals = False
    problemNodes = np.array(stripDatFiles("R101_0.5.dat"))
    #randomFactors = [1,5,10,15,20,25,30,35,40]
    randomFactors = [1,2,3,4,5]
    numberOfBatches = 5
    results = []
    for randomFactor in randomFactors:
        subresults = [randomFactor]
        for batch in range(numberOfBatches):
            subresults.append(main(problemNodes, randomFactor, temp, tempDecrease, visuals))
            print("Results for "+ str(randomFactor)+ " batch[" + str(batch)+"]")
        print("All results for "+ str(randomFactor))
        print(subresults)
        results.append(subresults)
    print("Hooray end!")
    print(results)

def testTwo():
    temp =  10000
    #tempDecreaseArray = [0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009,0.0001]
    tempDecreaseArray = [0.00003]
    visuals = True
    problemNodes = np.array(stripDatFiles("R101_0.5.dat"))
    #randomFactors = [1,5,10,15,20,25,30,35,40]
    randomFactor = 10
    numberOfBatches = 1
    results = []
    for tempDecrease in tempDecreaseArray:
        subresults = [randomFactor]
        for batch in range(numberOfBatches):
            subresults.append(main(problemNodes, randomFactor, temp, tempDecrease, visuals))
            print("Results for "+ str(randomFactor)+ " batch[" + str(batch)+"]")
        print("All results for "+ str(randomFactor))
        print(subresults)
        results.append(subresults)
    print("Hooray end!")
    print(results) 
#testOne()
testTwo()


