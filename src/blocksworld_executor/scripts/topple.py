#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# Solve a multi-commodity flow problem.  Two products ('Pencils' and 'Pens')
# are produced in 2 cities ('Detroit' and 'Denver') and must be sent to
# warehouses in 3 cities ('Boston', 'New York', and 'Seattle') to
# satisfy supply/demand ('inflow[h,i]').
#
# Flows on the transportation network must respect arc capacity constraints
# ('capacity[i,j]'). The objective is to minimize the sum of the arc
# transportation costs ('cost[i,j]').

import gurobipy as gp
from gurobipy import GRB
import sys
from functools import cmp_to_key
import re
from collections import defaultdict
import numpy as np

SINGLE_GOAL = 0
TIME = 150
MAX_HEIGHT = 4
#MAX_TOPPLE = 3
outfile = "out"
CANTOPPLE = True
if len(sys.argv)>1:
    CANTOPPLE = bool(int(sys.argv[1]))
    if CANTOPPLE:
        outfile = outfile+".topple"
    else:
        outfile = outfile+".notopple"
numo=0
if len(sys.argv)>2:
    numo = int(sys.argv[2])
SEED = 0
if len(sys.argv)>3:
    SEED = int(sys.argv[3])
NUMOBJ = numo
NUMLOC = min(8, NUMOBJ)
TIMESTEPS = NUMOBJ*4+3#*NUMOBJ
_infty = 100000
_small_infty = 100
_epsilon = .00001
_large_epsilon = 1
_extra = _epsilon#.1
_transfer_base_cost = 2
_topple_base_cost = 1

commodities = ["o"+str(i) for i in range(NUMOBJ)]
nodes = []
arcs = []
capacities = dict()
ncapacities = dict()
costs = dict()
actions = dict()
adjacencies = dict()
incidences = dict()

# Static pad layout (kept identical for every problem file you generate)
PAD_COORDS = {
    1: {'x': 0.47, 'y': -0.30},
    2: {'x': 0.47, 'y': -0.10},
    3: {'x': 0.47, 'y':  0.10},
    4: {'x': 0.47, 'y':  0.30},
    5: {'x': 0.30, 'y': -0.30},
    6: {'x': 0.30, 'y': -0.10},
    7: {'x': 0.30, 'y':  0.10},
    8: {'x': 0.30, 'y':  0.30},
}

def get_action(key):
    global actions
    if key in actions:
        return actions[key]
    return ""

def get_int_cost(cost):
    return int(np.ceil(cost))


def add_arc(u, v, capacity, cost, action):
    global arcs
    global capacities
    global costs
    global actions
    global adjacencies
    global incidences
    if (u,v) in arcs:
        print("Arc ",u, v, capacity, cost, action," already exists")
        return
    arcs.append((u,v))
    capacities[arcs[-1]] = capacity
    costs[arcs[-1]] = get_int_cost(cost)
    actions[arcs[-1]] = action
    adjacencies[u].append(v)
    incidences[v].append(u)

def get_inter_location_distance(lu, lv):
    # return 0
    if lu>=NUMLOC or lv>=NUMLOC:
        return 2
    else:
        ux = PAD_COORDS[lu+1]['x']
        uy = PAD_COORDS[lu+1]['y']
        vx = PAD_COORDS[lv+1]['x']
        vy = PAD_COORDS[lv+1]['y']
        dist = np.sqrt((ux-vx)*(ux-vx) + (uy-vy)*(uy-vy))
        return get_int_cost(np.ceil(np.sqrt((ux-vx)*(ux-vx) + (uy-vy)*(uy-vy)))) 
    # cost_to_go_between_locations = abs(lu-lv)
    # if lu>=NUMOBJ or lv>=NUMOBJ:
    #     cost_to_go_between_locations = 1 #table interaction cost
    # return cost_to_go_between_locations


transferarcs = dict()
starttopplearcs = dict()
looptopplearcs = dict()
for loc in range(NUMLOC):
    transferarcs[loc] = []
    starttopplearcs[loc] = []
    looptopplearcs[loc] = []

for loc in range(NUMLOC):
    #stack at location
    for stackpos in range(NUMOBJ):
        nodes.append("l_"+str(loc)+"_"+str(stackpos))
        ncapacities[nodes[-1]] = 1
        adjacencies[nodes[-1]] = []
        incidences[nodes[-1]] = []
        add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
        if(stackpos > 0):
            add_arc("l_"+str(loc)+"_"+str(stackpos),"l_"+str(loc)+"_"+str(stackpos-1),1,0,"")
            add_arc("l_"+str(loc)+"_"+str(stackpos-1),"l_"+str(loc)+"_"+str(stackpos),1,0,"")
    for loc2 in range(loc):
        add_arc("l_"+str(loc2)+"_"+str(NUMOBJ-1),"l_"+str(loc)+"_"+str(NUMOBJ-1),1, _transfer_base_cost*_small_infty + get_inter_location_distance(loc,loc2) ,"TRANSFER")
        transferarcs[loc].append(("l_"+str(loc2)+"_"+str(NUMOBJ-1),"l_"+str(loc)+"_"+str(NUMOBJ-1)))
        transferarcs[loc2].append(("l_"+str(loc2)+"_"+str(NUMOBJ-1),"l_"+str(loc)+"_"+str(NUMOBJ-1)))
        add_arc("l_"+str(loc)+"_"+str(NUMOBJ-1),"l_"+str(loc2)+"_"+str(NUMOBJ-1),1, _transfer_base_cost*_small_infty + get_inter_location_distance(loc,loc2) ,"TRANSFER")
        transferarcs[loc].append(("l_"+str(loc)+"_"+str(NUMOBJ-1),"l_"+str(loc2)+"_"+str(NUMOBJ-1)))
        transferarcs[loc2].append(("l_"+str(loc)+"_"+str(NUMOBJ-1),"l_"+str(loc2)+"_"+str(NUMOBJ-1)))

if CANTOPPLE:
    nodes.append("table")
    ncapacities[nodes[-1]] = NUMOBJ
    adjacencies[nodes[-1]] = []
    incidences[nodes[-1]] = []
    add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
    for loc in range(NUMLOC):
        nodes.append("p_"+str(loc)+"_"+str(NUMOBJ))
        ncapacities[nodes[-1]] = NUMOBJ
        adjacencies[nodes[-1]] = []
        incidences[nodes[-1]] = []
        add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
        # print(len(looptopplearcs[loc]), len(arcs))
        looptopplearcs[loc].append(arcs[-1])
        add_arc("l_"+str(loc)+"_"+str(NUMOBJ-1), nodes[-1], NUMOBJ, 0, "")
        starttopplearcs[loc].append(arcs[-1])
        add_arc(nodes[-1], "table", NUMOBJ, (_topple_base_cost+_extra)*_small_infty + 1, "TOPPLE")

    for obj in range(NUMOBJ):
        nodes.append("on_"+str(NUMLOC+obj)+"_0")
        ncapacities[nodes[-1]] = 1
        adjacencies[nodes[-1]] = []
        incidences[nodes[-1]] = []        
        add_arc(nodes[-1],nodes[-1],ncapacities[nodes[-1]],0,"")
        add_arc("table", nodes[-1], 1, 0, "")
        for loc in range(NUMLOC):
            # add_arc(nodes[-1], "l_"+str(loc)+"_"+str(NUMOBJ-1), 1, (_transfer_base_cost+_extra)*_small_infty + get_inter_location_distance(loc, NUMOBJ-1), "TRANSFER")
            add_arc(nodes[-1], "l_"+str(loc)+"_"+str(NUMOBJ-1), 1, get_int_cost((_transfer_base_cost+_extra)*_small_infty) + get_inter_location_distance(loc, NUMOBJ-1), "TRANSFER")


        
print("Nodes")
for node in nodes:
    print(node, "\n -- ", adjacencies[node], "\n -- ", incidences[node])
print("Arcs")
for arc in arcs:
    print(arc, capacities[arc], costs[arc], actions[arc])



#####################################################################

#####PROBLEM
problem = dict()
# problem["o0"] = ("l_0_0","l_1_0")
# problem["o1"] = ("l_0_1","l_1_1")
# problem["o2"] = ("l_0_2","l_1_2")

#BLOCKSWORLD
#Objects o0,o1,o2
#Locations
#     2  2   2       2
#     1  1   1       1
#     0  0   0       0
# l  l0  l1  l2 ... lk

# Start 
#    o2  2   2       2
#    o1  1   1       1
#    o0  0   0       0
# l  l0  l1  l2 ... lk

# Goal
#     2  o2  2       2
#     1  o1  1       1
#     0  o0  0       0
# l  l0  l1  l2 ... lk

if NUMOBJ == 9:
    COLOURS = ['Red1', 'Red2', 'Red3', 'Green1', 'Green2', 'Green3', 'Blue1', 'Blue2', 'Blue3']
    #COLOURS = ['Red1', 'Green1', 'Blue1', 'Green2', 'Blue2', 'Red2', 'Blue3', 'Red3', 'Green3']
elif NUMOBJ == 6:
    COLOURS = ['Red1', 'Red2', 'Red3', 'Green1', 'Green2', 'Green3']
elif NUMOBJ == 4:
    COLOURS = ['Red1', 'Red2', 'Green1', 'Green2']
else:
    COLOURS = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple', 'Cyan', 'Magenta']

import random
from collections import defaultdict
random.seed(SEED)
locindicesi = [0 for loc in range(NUMLOC)]
locindicese = [0 for loc in range(NUMLOC)]
'''
for c in commodities:
    startloc = random.randint(0,min(2,int(NUMLOC/2)-1))
    starth = locindicesi[startloc]
    locindicesi[startloc] = locindicesi[startloc]+1
    maxend = min(2,int(NUMLOC/2)-1)
    endloc = random.randint(0,maxend+1)
    if endloc == maxend+1:
        problem[c] = ("l_"+str(startloc)+"_"+str(starth),"")
    else: 
        endh = locindicese[endloc]
        locindicese[endloc] = locindicese[endloc]+1
        problem[c] = ("l_"+str(startloc)+"_"+str(starth),"l_"+str(endloc)+"_"+str(endh))
'''

'''
maxend = min(2, int(NUMLOC/2) - 1)  # unchanged from your code
flex_groups = defaultdict(list)      # endloc -> list of c needing fill later

for c in commodities:
    startloc = random.randint(0, min(2, int(NUMLOC/2) - 1))
    starth = locindicesi[startloc]
    locindicesi[startloc] += 1

    roll = random.randint(0, maxend + 1)  # same distribution as before
    if roll == maxend + 1:
        problem[c] = (f"l_{startloc}_{starth}", "")  # placeholder; fill later
        flex_groups[roll].append(c)
    else:
        endloc = roll
        endh = locindicese[endloc]
        locindicese[endloc] += 1
        problem[c] = (f"l_{startloc}_{starth}", f"l_{endloc}_{endh}")

# Finalize all deferred blocks: each group reserves contiguous heights
for endloc, cs in flex_groups.items():
    base = locindicese[endloc]         # current stack height at that end location
    k = len(cs)                              # number of deferred blocks in this group
    all_possible = ",".join(f"l_{endloc}_{h}" for h in range(base, base + k))
    for c in cs:
        start = problem[c][0]
        problem[c] = (start, all_possible)
    # Reserve those heights so future blocks see updated max height
    locindicese[endloc] += k
'''

'''
init_pool = list(range(NUMOBJ))
random.shuffle(init_pool)

problem = {}
for i in range(NUMOBJ):
    n = init_pool[i]
    init_label = f"l_{n//3}_{n%3}"
    goal_loc = 1 if "Red" in COLOURS[i] else 2 if "Green" in COLOURS[i] else 3
    goal_csv = ",".join(f"l_{goal_loc}_{h}" for h in range(3))
    problem[f"o{i}"] = (init_label, goal_csv)
'''

order = list(range(NUMOBJ))
random.shuffle(order)

stacks = [[], [], []]

if SINGLE_GOAL == 1:
    k = NUMOBJ // 2
    for n in order[:k]:
        stacks[0].append(n)
    for n in order[k:]:
        p = random.randrange(3)  # 0,1,2 with equal probability
        stacks[p].append(n)
else:
    k = 0
    for n in order[k:]:
        while True:
            p = random.randrange(3)  # 0,1,2 with equal probability
            if len(stacks[p]) > MAX_HEIGHT:
                continue
            stacks[p].append(n)
            break

pos = {}
for p in range(3):
    for h, n in enumerate(stacks[p]):  # h = height within stack p (bottom=0)
        pos[n] = (p, h)

problem = {}
P_ = min(NUMOBJ//2, 3)
gi = random.randint(0, NUMOBJ-1)
for i in range(NUMOBJ):
    p, h = pos[i]
    init_label = f"l_{p}_{h}"
    if SINGLE_GOAL == 1:
        if i == gi:
            goal_loc = "l_3_0"
        else:
            goal_loc = ""
    else:
        goal_loc = f"l_{i//P_+1}_{i%P_}"
    problem[f"o{i}"] = (init_label, goal_loc)

'''
problem["o0"] = ("l_0_0","l_2_0")#red1
problem["o1"] = ("l_0_1","")#green1
problem["o2"] = ("l_0_2","")#blue1
problem["o3"] = ("l_0_4","")#green2
problem["o4"] = ("l_1_0","l_1_0")#blue2
problem["o5"] = ("l_2_0","l_1_1")#red2
'''

#RGB
# update this based on the problem
#COLOURS = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple', 'Cyan', 'Magenta']

'''
problem["o0"] = ("l_0_0","l_0_0,l_0_1,l_0_2")#red1
problem["o1"] = ("l_0_1","l_1_0,l_1_1,l_1_2")#green1
problem["o2"] = ("l_0_2","l_2_0,l_2_1,l_2_2")#blue1
problem["o3"] = ("l_1_0","l_1_0,l_1_1,l_1_2")#green2
problem["o4"] = ("l_1_1","l_2_0,l_2_1,l_2_2")#blue2
problem["o5"] = ("l_1_2","l_0_0,l_0_1,l_0_2")#red2
problem["o6"] = ("l_2_0","l_2_0,l_2_1,l_2_2")#blue3
problem["o7"] = ("l_2_1","l_0_0,l_0_1,l_0_2")#red3
problem["o8"] = ("l_2_2","l_1_0,l_1_1,l_1_2")#green3
'''

'''
#COLOURS = ['cube_A', 'cube_N', 'cube_U', 'cube_C', 'cube_O', 'cube_M', 'cube_P', 'cube_2', 'cube_6']
COLOURS = ['cube_A', 'cube_N', 'cube_U', 'cube_C', 'cube_O', 'cube_M', 'cube_P']
problem["o0"] = ("l_1_0","l_1_2")#red1      'A'
problem["o1"] = ("l_0_0","l_1_1")#green1    'N'
problem["o2"] = ("l_2_2","l_1_0")#blue1     'U'
problem["o3"] = ("l_2_0","l_2_3")#green2    'C'
problem["o4"] = ("l_2_1","l_2_2")#blue2     'O'
problem["o5"] = ("l_0_1","l_2_1")#red2      'M'
problem["o6"] = ("l_0_2","l_2_0")#blue3     'P'
#problem["o7"] = ("l_2_1","l_2_1")#red3      '2'
#problem["o8"] = ("l_2_2","l_2_0")#green3    '6'
'''


# problem["o0"] = ("l_1_0","l_1_0")
# problem["o1"] = ("l_0_0","l_1_1")
# problem["o2"] = ("l_1_1","l_1_2")
# problem["o3"] = ("l_1_2","l_1_3")
'''

problem["o0"] = ("l_0_0","l_1_0")
problem["o1"] = ("l_0_1","")
problem["o2"] = ("l_0_2","")
problem["o3"] = ("l_0_3","")
'''


# problem["o0"] = ("l_0_0","l_1_0")
# problem["o1"] = ("l_0_1","l_1_1")
# problem["o2"] = ("l_0_2","l_1_2")
# problem["o3"] = ("l_0_3","l_1_3")
# problem["o4"] = ("l_0_4","l_1_4")





# problem["o0"] = ("l_0_0","l_1_0")
# problem["o1"] = ("l_0_1","l_1_2")
# problem["o2"] = ("l_0_2","l_2_0")
# problem["o3"] = ("l_0_3","l_3_0")

# problem["o0"] = ("l_0_0","l_1_0")
# problem["o1"] = ("l_0_1","l_1_1")
# problem["o2"] = ("l_0_2","l_1_2")
# problem["o0"] = ("l_0_0","table")
# problem["o1"] = ("l_0_1","table")
# problem["o2"] = ("l_0_2","table")
# problem["o0"] = ("l_0_0","on_0")
# problem["o1"] = ("l_0_1","on_1")
# problem["o2"] = ("l_0_2","on_2")
#####PROBLEM

for k, (start_pose, end_opts) in problem.items():
    print(f"{k}: start={start_pose} | end_options({len(end_opts)}): {end_opts}")

m = gp.Model("netflow")
modelvars = dict()
tcosts = dict()

for t in range(TIMESTEPS):
    for c in commodities:
        for arc in arcs:
            # modelvars[ (t,c,arc[0],arc[1]) ] = m.addVar(lb=0.0, ub=capacities[arc], vtype=GRB.INTEGER, name=str(t)+str(c)+str(arc[0])+str(arc[1]))
            inflation = 1#_infty*(t+1)
            cost = costs[arc]
            if cost == 0:
                cost = _epsilon
            if arc[0]==arc[1]:
                inflation = 1
                cost = 0
            tcosts[t,arc] = cost*inflation
            modelvars[ (t,c,arc[0],arc[1]) ] = m.addVar(lb=0.0, ub=capacities[arc], obj=cost*inflation, vtype=GRB.INTEGER, name=str(t)+str(c)+str(arc[0])+str(arc[1]))



for t in range(TIMESTEPS):
    for arc in arcs:
        #each edge allows at most capacity flow
        m.addConstr(gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities) <= capacities[arc])
        pass


for t in range(TIMESTEPS):
    for node in nodes:
        #each node allows at most capacity of node to flow out
        m.addConstr(  gp.quicksum( modelvars[t,c,node,nout] for nout in adjacencies[node] for c in commodities ) <= ncapacities[node] )
        # if t>0:
        m.addConstr(  gp.quicksum( modelvars[t,c,nin,node] for nin in incidences[node] for c in commodities ) <= ncapacities[node] )
        pass

    already_crossed = []
    for arc in arcs:
        rarc = (arc[1], arc[0])
        if arc[0] == arc[1]:
            continue
        if arc not in already_crossed and rarc in arcs:
            if capacities[arc] != capacities[rarc]:
                print("Forward and reverse edges need to have the same capacity")
            m.addConstr(  gp.quicksum( modelvars[t,c,arc[0],arc[1]] for c in commodities ) +  gp.quicksum( modelvars[t,c,rarc[0],rarc[1]] for c in commodities ) <= capacities[arc] )
            already_crossed.append(arc)
            already_crossed.append(rarc)

    if t==0:
        continue

    for c in commodities:
        for node in nodes:
            #flow
            m.addConstr(  gp.quicksum( modelvars[t-1,c,nin,node] for nin in incidences[node] ) == gp.quicksum( modelvars[t,c,node,nout] for nout in adjacencies[node] )  )
            pass


loopback = []

for c in commodities:
    # loop = m.addVar(lb=0.0, ub=1, obj=0, vtype=GRB.INTEGER, name="-1"+str(c)+str(-1)+str(-1))
    loop = m.addVar(lb=0.0, ub=1, vtype=GRB.INTEGER, name="-1"+str(c)+str(-1)+str(-1))
    loopback.append(loop)
    modelvars[-1,c,-1,-1] = loop
    costs[(-1,-1)] = 0
    capacities[(-1,-1)] = 1


    m.addConstr( loop == 1 )

    start = problem[c][0]#"l_"+str(problem[c][0][0])+"_"+str(problem[c][0][1])
    print("start", start)
    m.addConstr( loop == gp.quicksum( modelvars[0,c,start,nout] for nout in adjacencies[start] ) )

    for arc in arcs:
        if arc[0] != start or arc[1] not in adjacencies[start]:
            # print(0,c,arc, "set to 0")
            m.addConstr( modelvars[0,c,arc[0],arc[1]] == 0)


    #changing to accept multiple goals
    end = problem[c][1]#"l_"+str(problem[c][1][0])+"_"+str(problem[c][1][1])
    print("end", end)
    if end != "":
        if "," not in end:
            m.addConstr( gp.quicksum( modelvars[TIMESTEPS-1,c,nin,end] for nin in incidences[end] ) == loop )
            for arc in arcs:
                if arc[1] != end or arc[0] not in incidences[end]:
                    m.addConstr( modelvars[TIMESTEPS-1,c,arc[0],arc[1]] == 0)
        else:
            ends_vec = end.split(",")
            all_incidences = []
            for n in ends_vec:
                if len(n.split("_"))==3:
                    if int(n.split("_")[2])<NUMOBJ:
                        for nn in incidences[n]:
                            all_incidences.append((nn, n))
            m.addConstr( gp.quicksum( modelvars[TIMESTEPS-1,c,nin[0],nin[1]] for nin in all_incidences ) == loop )
    else:
        all_incidences = []
        for n in nodes:
            if len(n.split("_"))==3:
                if int(n.split("_")[2])<NUMOBJ:
                    for nn in incidences[n]:
                        all_incidences.append((nn, n))
        m.addConstr( gp.quicksum( modelvars[TIMESTEPS-1,c,nin[0],nin[1]] for nin in all_incidences ) == loop )



    #DEBUG
    # m.addConstr( modelvars[3, 'o2', 'p_0_3', 'table']==1 )
    # m.addConstr( modelvars[3, 'o1', 'p_0_3', 'table']==1 )
    # m.addConstr( modelvars[3, 'o0', 'p_0_3', 'table']==1 )
    # m.addConstr( modelvars[2, 'o2', 'l_0_3', 'l_1_3']==0 )



####Indicators
indicatorvals = dict()
for t in range(TIMESTEPS):
    for arc in arcs:
        M = NUMOBJ+1
        indicatorvals[t,arc[0],arc[1]] = m.addVar( vtype=GRB.BINARY)
        if capacities[arc] == 1:
            m.addConstr(indicatorvals[t,arc[0],arc[1]] == gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities))
            # gp.quicksum(ex[u,v] for (u,v) in worlds_to_edges[w]) + E*(1-wx[w]) <= E
        else:
            m.addConstr( gp.quicksum(modelvars[t,c,arc[0],arc[1]] for c in commodities) + M*(1-indicatorvals[t,arc[0],arc[1]]) <= M )

####TOPPLEFIX
for t in range(TIMESTEPS):
    for loc in range(NUMLOC):
        m.addConstr(gp.quicksum(indicatorvals[t, arc[0], arc[1]]  for arc in transferarcs[loc]) + gp.quicksum(indicatorvals[t, arc[0], arc[1]]  for arc in starttopplearcs[loc]) <= 1 )
        m.addConstr(gp.quicksum(indicatorvals[t, arc[0], arc[1]]  for arc in transferarcs[loc]) + gp.quicksum(indicatorvals[t, arc[0], arc[1]]  for arc in looptopplearcs[loc]) <= 1 )

m.setObjective( gp.quicksum( indicatorvals[t,arc[0],arc[1]]*tcosts[t,arc] for arc in arcs for t in range(TIMESTEPS) ), GRB.MINIMIZE )
# m.setObjective( gp.quicksum( modelvars[t,c,arc[0],arc[1]]*costs[arc] for t in range(TIMESTEPS) for c in commodities for arc in arcs ), GRB.MINIMIZE )
# m.setObjective( gp.quicksum(gp.quicksum( modelvars[t,c,arc[0],arc[1]] for c in commodities )*costs[arc] for arc in arcs for t in range(TIMESTEPS) ), GRB.MINIMIZE )
m.setParam(GRB.Param.TimeLimit, TIME)
m.optimize()


def print_problem(locindices, prob, index):
    global commodities
    for l, oheight in enumerate(locindices):
        objseq = ["x" for h in range(oheight)]
        # for h in range(oheight):
        # objstr = "x"
        for c in commodities:
            state = prob[c][index]
            try:
                # print("trying ",c,l," with ",state.split("_")[1],state.split("_")[2])
                if int(state.split("_")[1])==l:
                    objstr = objseq[int(state.split("_")[2])] = c
                    # break
            except:
                pass
        if oheight==0:
            objseq = ["x"]
        # print(l,oheight,objseq)
        joinstr = "|".join(objseq)
        print("l"+str(l)+": "+joinstr)

# for c in commodities:
#     print(c, ": ", problem[c])
# print("PROBLEM ",CANTOPPLE,NUMOBJ,SEED)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Start>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print_problem(locindicesi, problem, 0)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Goal>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# print_problem(locindicese, problem, 1)
# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
def locindex(ac):
    u = ac[0]
    v = ac[1]
    lint = -1
    hint = -1
    try:
        lint = int(u.split("_")[1])
        hint = int(u.split("_")[2])
    except:
        pass
    lend = -1
    hend = -1
    try:
        lend = int(v.split("_")[1])
        hend = int(v.split("_")[2])
    except:
        pass
    return (lint,hint,lend,hend)
def compare (act1, act2):
    uv1 = locindex(act1)
    uv2 = locindex(act2)
    if uv1[0] == uv2 [0]: #same stack location so order by height
        if uv1[1] < uv2[1]:
            return 1
        if uv1[1] > uv2[1]:
            return -1
        if uv1[1] == uv2[1]:
            return 0
    elif uv1[2] == uv2[0]: #act 1 target stack matches act 2 source stack. do act 2 first
        return 1
    elif uv1[0] == uv2[2]:  #act 1 source stack matches act 2 target stack. do act 1 first
        return -1
    else: #any order should work. sort by stack location index.
        if uv1[0] < uv2[0]:
            return 1
        if uv1[0] > uv2[0]:
            return -1
        if uv1[0] == uv2[0]:
            return 0
def cmp_to_key(mycmp):
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K
# Print solution
# if m.Status == GRB.OPTIMAL:
if m.SolCount > 0:
    # solution = {v.varName: v.X for v in m.getVars() if math.abs(v.X) > epsilon}
    print("------------------")
    for t in range(TIMESTEPS):#= 1#TIMESTEPS-1
        for c in commodities:
            for arc in arcs:
                var = modelvars[ (t,c,arc[0],arc[1]) ]
                if True:#var.X > 0:
                    action = get_action(arc)
                    # print((t,c,arc[0],arc[1]), action, capacities[arc], costs[arc], var.X)
    print("------------------")
    for var in modelvars:
        if modelvars[var].X > 0:
            obj = var[0]
            u = var[2]
            v = var[3]
            action = get_action((u,v))
            # print(var, modelvars[var].X, action, capacities[(u,v)], costs[(u,v)])
            # if action != "":
            #     print(var, action, modelvars[var].X, costs[(u,v)])
    print("------------------")
    solvars = []
    solactions = dict()
    for t in range(TIMESTEPS):
        solactions[t] = []
    for var in modelvars:
        if modelvars[var].X > _epsilon:
            obj = var[1]
            u = var[2]
            v = var[3]
            action = get_action((u,v))
            # print(var, modelvars[var].X, action, capacities[(u,v)], costs[(u,v)])
            if action != "":
                solvars.append(var)
                lint = int(u.split("_")[1])
                hint = int(u.split("_")[2])
                print(var, action, modelvars[var].X, costs[(u,v)])
                solactions[var[0]].append((lint,hint,var))
    print("#####SOLUTION")

    TOT_COST = 0
    for t in range(TIMESTEPS):
        # print("t=",t)
        currentaction = dict()
        for act in solactions[t]:
            var = act[2]
            obj = var[1]
            u = var[2]
            v = var[3]
            if (u,v) not in currentaction.keys():
                source = u
                sink = v
                try:
                    source = "l"+str(u.split("_")[1])
                except:
                    pass
                try:
                    sink = "l"+str(v.split("_")[1])
                except:
                    pass
                currentaction[(u,v)] = [get_action((u,v))+" "+source+"->"+sink+" : "]
            else:
                if not CANTOPPLE:
                    print("!!!Unexpected: This should never happen.",var)
                else:
                    #print("Aggregating grouped actions.")
                    pass
            currentaction[(u,v)].append(obj)
        
        keys = [k for k in currentaction.keys()]
        keys.sort(key=cmp_to_key(compare))
        for key in keys:
            print("".join(currentaction[key]))
    solcost = int(m.getObjective().getValue())
    print("Objective: ",solcost)
    # f = open(outfile, "a")
    # f.write(str(solcost)+"\n")
    # f.close()


import re
import yaml
from functools import cmp_to_key

# ---------------------------------------------------------------------------
OBJ_TO_COLOUR = {f"o{i}": colour for i, colour in enumerate(COLOURS)}

# ---------------------------------------------------------------------------
def _rebuild_solution_lines(solactions, TIMESTEPS):
    """
    Reconstruct human-readable action lines with pos+height preserved.

    • TRANSFER/MOVE prints both position and height, e.g. l0_5->l2_5
    • If a move originates from a toppled-on-table location like 'on_8_0',
      we parse it and print as l8_0 (using the pos+height from the token).
    • TOPPLE prints src with pos+height and dst 'table' (objects concatenated).
    """
    lines = []

    def parse_loc(tok):
        """Return (kind, pos, h) or ('table', None, None) / (tok, None, None)."""
        if tok == "table":
            return ("table", None, None)
        m = re.match(r"^([A-Za-z]+)_(\-?\d+)_([\-]?\d+)$", tok)
        if not m:
            return (tok, None, None)
        kind, pos, h = m.group(1), int(m.group(2)), int(m.group(3))
        return (kind, pos, h)

    def pretty(tok):
        """
        Normalize any location-like token (l_*, p_*, on_*) to 'l{pos}_{h}'.
        'table' stays 'table'. Unknown tokens are returned as-is.
        """
        kind, pos, h = parse_loc(tok)
        if kind == "table":
            return "table"
        if pos is None or h is None:
            return tok
        return f"l{pos}_{h}"

    for t in range(TIMESTEPS):
        # Group all actions by (u, v) in this timestep so objects to same sink are concatenated.
        groups = {}  # (u, v) -> [obj, obj, ...]
        for act in solactions.get(t, []):
            _, _, var = act
            obj, u, v = var[1], var[2], var[3]
            groups.setdefault((u, v), []).append(obj)

        # Emit using your existing ordering
        for (u, v) in sorted(groups.keys(), key=cmp_to_key(compare)):
            objs = groups[(u, v)]
            action_name = get_action((u, v))

            src = pretty(u)
            dst = pretty(v)

            if v == "table":
                # TOPPLE: concatenate objects in order
                lines.append(f"{action_name} {src}->{dst} : " + "".join(objs))
            else:
                # TRANSFER/MOVE: src may be l_*_* / p_*_* / on_*_*;
                # pretty() ensures we print pos+height (e.g., on_8_0 -> l8_0)
                lines.append(f"{action_name} {src}->{dst} : " + "".join(objs))

    return lines

# ---------------------------------------------------------------------------
def _yaml_inline_block(d):
    """Return '{key1: v1, key2: v2, ...}' with nice spacing."""
    return '{' + ', '.join(f'{k}: {v}' for k, v in d.items()) + '}'

def _dump_manual_yaml(file_name, pads, initial, goal, actions):
    lines = []

    # ---------- pads --------------------------------------------------------
    lines.append('pads: ')
    for idx in sorted(pads):
        p = pads[idx]
        lines.append(
            f'  {idx}: {{x: {p["x"]:6.2f}, y: {p["y"]:6.2f}}}'
        )

    # ---------- constants ---------------------------------------------------
    lines += [
        '',
        'z_base:      0.025    # table + 1/2 block',
        'z_increment: 0.05     # full block height',
        '# ----------------------------------------------------------',
        ''
    ]

    # ---------- initial -----------------------------------------------------
    lines.append('initial:')
    for item in initial:
        # keep same order: name → pad → height
        entry = _yaml_inline_block(
            {'name': item["name"], 'pad': item["pad"], 'height': item["height"]}
        )
        lines.append(f'  - {entry}')
    lines.append('')

    # ---------- goal --------------------------------------------------------
    lines.append('goal:')
    for item in goal:
        entry = _yaml_inline_block(
            {'name': item["name"], 'pad': item["pad"], 'height': item["height"]}
        )
        lines.append(f'  - {entry}')
    lines.append('')

    # ---------- actions -----------------------------------------------------
    lines.append('actions:     # ordered action sequence')
    for act in actions:
        print(act)
        lines.append(f'  - type:  {act["type"]}')

        if act['type'] == 'Smash':
            seq = act.get('blocks')
            lines.append(f'    blocks: [{", ".join(seq)}]')
            lines.append('    from: ' + _yaml_inline_block(act['from']))
            lines.append('    dir:   +x')
        else:                         # Move
            # Move may have both 'from' and 'to' (we only need 'to' per spec)
            lines.append('    block: ' + act['block'])
            if 'from' in act:         # include 'from' only if user wants it
                lines.append('    from: ' + _yaml_inline_block(act['from']))
            lines.append('    to:   ' + _yaml_inline_block(act['to']))
        lines.append('')              # blank line between actions

    # ---------- write -------------------------------------------------------
    with open(file_name, 'w') as fh:
        fh.write('\n'.join(lines))

# ---------------------------------------------------------------------------
def convert_to_yaml(problem, solactions, TIMESTEPS, file_name='problem.yaml'):
    """YAML writer that understands table positions l# ≥ n_objs."""

    n_objs      = len(problem)
    total_pads  = 2 * n_objs                     # 0…n_objs-1 = real pads
                                                 # n_objs…2*n_objs-1 = table
    # ------------------------------------------------------------------ #
    # 1) Pad stacks – prepare every pad index up-front (all start empty)
    pad_stacks  = {i: [] for i in range(total_pads)}

    initial_yml = []
    pad_buckets = defaultdict(list)   # pad -> list of (height, obj)

    #print(problem)
    for obj, (init, _) in problem.items():
        pad, h = map(int, init.split('_')[1:])   # 'l_{pad}_{h}'
        pad_buckets[pad].append((h, obj))
        initial_yml.append({'name': OBJ_TO_COLOUR[obj], 'pad': pad + 1, 'height': h})

    # Show unsorted grouping (as inserted from problem)
    stack0 = {p: [obj for _, obj in pad_buckets.get(p, [])] for p in range(total_pads)}
    #print("stack0:", stack0)

    # Order stacks correctly as TOP→BOTTOM (highest height first)
    for p in range(total_pads):
        items = pad_buckets.get(p, [])
        pad_stacks[p] = [obj for h, obj in sorted(items, key=lambda t: t[0], reverse=True)]

    #print("stack1:", pad_stacks)

    # ------------------------------------------------------------------ #
    # 2) Goal section
    goal_yml = []
    for obj, (_, goals) in problem.items():
        if goals:                                           # at least one goal given
            # split on commas, drop empty fragments, keep order
            goal_tokens = [g.strip() for g in goals.split(',') if g.strip()]

            pads, heights = [], []
            for g in goal_tokens:
                gpad, gh = map(int, g.split('_')[1:])       # e.g. "l_0_2" -> (0, 2)
                pads.append(gpad + 1)                       # convert pad 0-based → 1-based
                heights.append(gh)

            # build YAML entry: scalars for 1 goal, lists for many
            entry = {
                'name': OBJ_TO_COLOUR[obj],
                'pad':    pads[0]   if len(pads) == 1 else pads,
                'height': heights[0] if len(heights) == 1 else heights,
            }
            goal_yml.append(entry)

        else:                                               # “don’t care”
            goal_yml.append({
                'name': OBJ_TO_COLOUR[obj],
                'pad': '-1',
                'height': '-1',
            })

    # ------------------------------------------------------------------ #
    # 3) Parse solver lines and build action list
    sol_lines   = _rebuild_solution_lines(solactions, TIMESTEPS)
    print("rebuilt sol lines:", sol_lines)

    # Lines look like:
    #   TOPPLE   lS_H->table : o0o4
    #   TRANSFER lS_H->lD_H  : o4
    #   MOVE     lS_H->lD_H  : o1
    #
    # pad_stacks example (TOP→BOTTOM):
    #   {0: ['o1','o2','o3'], 1: ['o0','o4','o5'], ...}
    # heights are: h = len(stack) - 1 - index
    #
    # We TRUST pad positions from solactions, but IGNORE heights there.
    # Heights come from (and update) pad_stacks using pop/insert directly.

    re_topple = re.compile(r'^TOPPLE\s+l(-?\d+)_(-?\d+)->table\s*:\s*((?:o\d+)+)$')
    re_move   = re.compile(r'^(?:TRANSFER|MOVE)\s+l(-?\d+)_(-?\d+)->l(-?\d+)_(-?\d+)\s*:\s*(o\d+)$')

    actions_yml = []

    for line in sol_lines:
        #print("stacks:", pad_stacks)
        #print("line:", line)
        line = line.strip()

        # ---------- TRANSFER / MOVE (single object) -----------------------------
        m = re_move.match(line)
        if m:
            s_pad = int(m.group(1))   # trust pad position, ignore s_h from line
            d_pad = int(m.group(3))   # trust pad position, ignore d_h from line
            obj   = m.group(5)

            # ensure pads exist
            pad_stacks.setdefault(s_pad, [])
            pad_stacks.setdefault(d_pad, [])

            # source height from pad_stacks; pop from actual location
            if obj in pad_stacks[s_pad]:
                i_src = pad_stacks[s_pad].index(obj)                 # index in TOP→BOTTOM
                h_src = len(pad_stacks[s_pad]) - 1 - i_src           # height from bottom
                pad_stacks[s_pad].pop(i_src)                         # remove from source
            else:
                # object is on table (from prior TOPPLE) or missing; treat as height 0
                h_src = 0

            # destination height is current size before placement
            h_dst = len(pad_stacks[d_pad])

            # place on top of destination (TOP→BOTTOM representation)
            pad_stacks[d_pad].insert(0, obj)

            actions_yml.append({
                'block': OBJ_TO_COLOUR[obj],
                'type' : 'Move',
                'from' : {'pad': s_pad + 1, 'height': h_src},
                'to'   : {'pad': d_pad + 1, 'height': h_dst},
            })
            continue

        # ---------- TOPPLE (possibly multiple objects) --------------------------
        m = re_topple.match(line)
        if m:
            s_pad = int(m.group(1))   # trust pad position, ignore s_h from line
            objs  = re.findall(r'o\d+', m.group(3))   # solver-provided order

            # 1) Check all objs are on the same pad and that pad == s_pad
            obj_pad_map = {}
            for o in objs:
                found_pad = None
                for p, stk in pad_stacks.items():
                    if o in stk:
                        found_pad = p
                        break
                obj_pad_map[o] = found_pad

            pads_present = set(v for v in obj_pad_map.values() if v is not None)
            if (None in obj_pad_map.values()) or len(pads_present) != 1 or (s_pad not in pads_present):
                print(f"[ERROR] TOPPLE objects must all be on pad {s_pad}; got {obj_pad_map}")
                sys.exit(-10)

            # 2) Verify the last listed object is actually the bottom one in the stack
            stack = pad_stacks[s_pad]  # TOP→BOTTOM
            idx_map = {o: stack.index(o) for o in objs}         # obj -> index in TOP→BOTTOM
            bottom_in_stack = max(idx_map.items(), key=lambda kv: kv[1])[0]  # largest index = lowest
            if objs[-1] != bottom_in_stack:
                print(f"[WARN] Last object in TOPPLE line ({objs[-1]}) is not bottom in stack (bottom is {bottom_in_stack}).")

            # 3) Assign the true bottom object
            bottom_obj = bottom_in_stack

            # compute bottom_obj height from current stack and strip tower above it (incl. itself)
            i_bot    = stack.index(bottom_obj)                 # index in TOP→BOTTOM
            tower_top_to_bottom = stack[:i_bot + 1]     # top..bottom (inclusive)
            tower_bottom_to_top = list(reversed(tower_top_to_bottom))  # bottom..top
            h_bottom = len(stack) - 1 - i_bot
            # keep only what's below the toppled tower
            pad_stacks[s_pad] = stack[i_bot + 1:]

            actions_yml.append({
                'blocks': [OBJ_TO_COLOUR[o] for o in tower_bottom_to_top],
                'type'  : 'Smash',
                'from'  : {'pad': s_pad + 1, 'height': h_bottom},
                'dir'   : '+x',
            })

            # toppled blocks lie on the table; do NOT assign them to any pad here
            continue

        raise ValueError(f"Un-recognised action line:\n  {line}")

    print("stacks:", pad_stacks)

    # ------------------------------------------------------------------ #
    # 4) Assemble YAML
    data = {
        'pads'        : PAD_COORDS,
        'z_base'      : 0.025,
        'z_increment' : 0.05,
        'initial'     : initial_yml,
        'goal'        : goal_yml,
        'actions'     : actions_yml,
    }

    _dump_manual_yaml(file_name, PAD_COORDS,
                      initial_yml, goal_yml, actions_yml)

    print(f"[OK] problem description written to '{file_name}'")

# ---------------------------------------------------------------------------
SDF_HEADER = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="blocks_world">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin
      filename="ignition-gazebo-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
      filename="ignition-gazebo-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin
      filename="ignition-gazebo-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin
      filename="ignition-gazebo-contact-system"
      name="gz::sim::systems::Contact">
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
"""

SDF_FOOTER = """  </world>
</sdf>
"""

def _format_float(val, two_dp=False):
    """Return a string with either two decimals (for x, y) or
       trimmed decimals (for z)."""
    if two_dp:
        return f"{val:0.3f}"
    s = f"{val:0.3f}".rstrip('0').rstrip('.')
    # ensure at least one decimal digit, e.g. '0' -> '0.0'
    return s if '.' in s else s + '.0'

def generate_sdf(problem,
                 pad_coords=PAD_COORDS,
                 z_increment=0.05,
                 file_name='blocks_world.sdf'):
    """
    Create an SDF file whose variable part is the <include> blocks
    derived from `problem`'s initial literals.
    """
    # ----- build <include> blocks -----------------------------------------
    include_blocks = []
    for obj, (init_state, _) in problem.items():
        pad_idx, height = map(int, init_state.split('_')[1:])

        # Convert pad (0-based in problem) to pad key (1-based in PAD_COORDS)
        pad_key = pad_idx + 1
        coords  = pad_coords[pad_key]
        x_str   = _format_float(coords['x'], two_dp=True)
        y_str   = _format_float(coords['y'], two_dp=True)
        z_str   = _format_float(height * z_increment)

        colour  = OBJ_TO_COLOUR[obj].lower()
        include_blocks.append(
            f"""    <include>
        <uri>model://wood_cube_{colour}</uri>
        <static>false</static>
        <pose>{x_str} {y_str} {z_str} 0 0 0</pose>
    </include>"""
        )

    # ----- write file ------------------------------------------------------
    with open(file_name, 'w') as fh:
        fh.write(SDF_HEADER + "\n")
        fh.write("\n".join(include_blocks) + "\n\n")
        fh.write(SDF_FOOTER)

    print(f"[OK] SDF scene written to '{file_name}'")

# ---------------------------------------------------------------------------
convert_to_yaml(problem, solactions, TIMESTEPS, 'my_problem.yaml')
generate_sdf(problem, file_name='my_blocks_world.sdf')

