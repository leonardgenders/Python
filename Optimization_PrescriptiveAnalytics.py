# ## Model Formulation
# 
# **Sets** \
# $C$: set of computers \{1=Basic, 2=XP, 3=VXP\} \
# $P$: set of processes \{1=assembly, 2=testing\}
# 
# **Parameters** \
# $v$: vector of unit profits ($v_1 = 80$, $v_2 = 129$, $v_3 = 152$) \
# $d$: vector of demands ($d_1 = 600$, $d_2 = 1200$, $d_3 = 50$) \
# $r$: vector of available hourly resources ($r_1 = 10000$, $r_2 = 3000$) \
# $A$: matrix of hourly resource requirements, where $a_{i,i}$ represents the hours of process $i$ to produce computer $j$. For example, $a_{1,2} = $6 because it takes $6$ hours of assembly ($i=1$) to produce XP computers ($j=2$).
# 
# **Decision Variable** \
# $x$: vector amount of each type of computer to produce, where $x_1$ is Basic, $x_2$ is XP, and $x_3$ is VXP.
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\max} & 80x_1 + 129x_2 + 152x_3 &\\
# \textrm{s.t.} & 5x_1 + 6x_2 + 8x_3 & \leq & 10000 & \\
# & x_1 + 2x_2 + 3x_3 & \leq & 3000 & \\
# & x_1 & \leq & 600 \\
# & x_2 & \leq & 1200 \\
# & x_3 & \leq & 50 \\
# & x_1, x_2, x_3 & \geq & 0 \\
# \end{matrix}
# \end{equation*}
# 
# 

# ## Python/Gurobi Model
# I am using the lists with 'Brute Force' Approach from Insights Video 2.3

# In[3]:


from gurobipy import *
m = Model('Problem3_2a')

# Sets
# Not defining sets - using 'brute force' method

# Parameters
# Not defining parameters - using 'brute force' method

# Decision Variables
## Number basic computers to produce
x1 = m.addVar(name='Basic', ub=600, lb=0)
## Number XP computers to produce
x2 = m.addVar(name='XP', ub=1200, lb=0)
## Number VXP computers to produce
x3 = m.addVar(name='VXP', ub=50, lb=0)

# Objective Function
## Maximize total profit
m.setObjective(80*x1 + 129*x2 + 152*x3, GRB.MAXIMIZE)
m.update()

# Constraints
## Assembly hours constraint
m.addConstr(5*x1 + 6*x2 + 8*x3 <= 10000)
## Testing hours constraint
m.addConstr(x1 + 2*x2 + 3*x3 <= 3000)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
print("PC Tech should produce %s units of Basic computers." % round(x1.x))
print("PC Tech should produce %s units of XP computers." % round(x2.x))
print("PC Tech should produce %s units of VXP computers." % round(x3.x))
print("The total profit is %s." % round(m.objval, 2))


# ## Sensitivity Analysis
# Perform sensitivity analysis in Python by iterating over the specified values and resolving your model each time.\
# Changing the selling price of VXP computers from $500$ to $650$ in increments of $10$.
# 

# In[6]:


from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
m = Model('Problem3_2a')

# Setup the model
# Decision Variables
## Number basic computers to produce
x1 = m.addVar(name='Basic', ub=600, lb=0)
## Number XP computers to produce
x2 = m.addVar(name='XP', ub=1200, lb=0)
## Number VXP computers to produce
x3 = m.addVar(name='VXP', ub=50, lb=0)

# Objective Function
## Maximize total profit
m.setObjective(80*x1 + 129*x2 + (560-275-(11*8)-(15*3))*x3, GRB.MAXIMIZE)
m.update()

# Constraints
## Note that upper and lower bound constraints on decision variables are addressed within variable declarations abvoe.
## Assembly hours constraint
m.addConstr(5*x1 + 6*x2 + 8*x3 <= 10000)
## Testing hours constraint
m.addConstr(x1 + 2*x2 + 3*x3 <= 3000)
m.update()

# define data structure to hold values of interest
df = pd.DataFrame(columns=['PriceVXP', 'NumBasic', 'NumXP', 'NumVXP', 'Profit'])
svxp = np.arange(500, 651, 10)

# loop over changing values
for i in range(len(svxp)):
    # update and solve
    m.setObjective(80*x1 + 129*x2 + (svxp[i]-275-(11*8)-(15*3))*x3, GRB.MAXIMIZE)
    m.update()
    m.optimize()
    
    df.loc[i] = [svxp[i], x1.x, x2.x, x3.x, m.objVal]

# view results
df


# Create a graph of the objective function value vs. the VXP selling price.

# In[7]:


plt.plot(df['PriceVXP'], df['Profit'])
plt.title('Sensitivity of Profit Changing to VXP Price')
plt.xlabel('VXP Price')
plt.ylabel('Profit')
plt.show()



# ## Modified Brick Problem
# Consider Brick product mix problem discussed in Module 1. Modify this problem by adding a third product: Coffee Tables. A single Coffee Table generates $15 in profit, uses 2 Big Blocks, and uses 1 Small Block. Use the same data for Tables and Chairs as stated in the original problem.

# Screenshot of Excel model for BRICK Problem

# In[11]:


from IPython import display
display.Image("BRICK_model.png")


# Screenshot of Excel Solver for BRICK Problem

# In[12]:


from IPython import display
display.Image("BRICK_solver.png")


# ## Model Formulation - Modified BRICK Problem
# 
# **Sets** \
# $L$: set of BRICK builds \{1=Table, 2=Chair, 3=CoffeeTable\} \
# $B$: set of avaialble blocks\{1=Big, 2=Small\} \
# 
# **Parameters** \
# $v$: vector of unit profits ($v_1 = 16$, $v_2 = 10$, $v_3 = 15$) \
# $r$: vector of available block resources ($r_1 = 12$, $r_2 = 18$), where $r_1$ represents big blocks and $r_2$ represents small blocks \
# $A$: matrix of block resource requirements, where $a_{i,j}$ represents the amount of blocks $i$ to produce the BRICK build $j$. For example, $a_{1,2} = $1 because it takes $1$ Big Block ($i=1$) to produce one Chair ($j=2$).
# 
# **Decision Variable** \
# $x$: vector amount of each type of BRICK build to produce, where $x_1$ is Table, $x_2$ is Chair, and $x_3$ is CoffeeTable.
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\max} & 16x_1 + 10x_2 + 15x_3 &\\
# \textrm{s.t.} & 2x_1 + x_2 + 2x_3 & \leq & 12 & \\
# & 2x_1 + 2x_2 + x_3 & \leq & 18 & \\
# & x_1, x_2, x_3 & \geq & 0 \\
# \end{matrix}
# \end{equation*}

# ## Python/Gurobi Model - Modified BRICK Problem
# I am using the lists with 'Brute Force' Approach from Insights Video 2.3

# In[13]:


from gurobipy import *
m = Model('ModBRICK')

# Sets
# Not defining sets - using 'brute force' method

# Parameters
# Not defining parameters - using 'brute force' method

# Decision Variables
## Number Tables to produce
x1 = m.addVar(name='Table', lb=0)
## Number Chairs to produce
x2 = m.addVar(name='Chair', lb=0)
## Number Coffee Tables to produce
x3 = m.addVar(name='CoffeeTable', lb=0)

# Objective Function
## Maximize total profit
m.setObjective(16*x1 + 10*x2 + 15*x3, GRB.MAXIMIZE)
m.update()

# Constraints
## Big Block constraint
m.addConstr(2*x1 + 1*x2 + 2*x3 <= 12)
## Small Block constraint
m.addConstr(2*x1 + 2*x2 + x3 <= 18)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
print("I should produce %s units of Tables." % round(x1.x))
print("I should produce %s units of Chairs." % round(x2.x))
print("I should produce %s units of Coffee Tables." % round(x3.x))
print("The total profit is %s." % round(m.objval, 2))


# ## 1) Model Formulation - Employee Scheduling
# 
# **Sets** \
# $D$: set of days \{0=Monday, 1=Tuesday, ..., 6=Sunday\}
# 
# **Parameters** \
# $r$: vector minimum number of employees required each day ($r_0$ = 7, $r_1$ = 5, ...) \
# $A$: workday adjacency matrix \
# Additional consideration: All employees are scheduled for 4-day work weeks
# 
# **Decision Variable** \
# $x_i$: number of employees that begin their 4-day work week on day $i$
# 
# **Objective Function and Constraints** \
# Goal: Minimize the number of employees that are working throughout the week \
# The optimization model is formulated as \
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\min} & \underset{j \in D}{\sum} x_j &\\
# \textrm{s.t.} & \underset{j \in D}{\sum}a_{i,j}x_j & \geq & r_i & \forall i \in D \\
# & x_j & \geq & 0 & \forall j \in D \\
# \end{matrix}
# \end{equation*}

# sum of ai,j xj is number of employees working on day i \
# r_i (number of employees required on day i)

# ## Screenshot of Excel Model - Lifeguard Scheduling


from IPython import display
display.Image("lifeguard_model.png")


# From our Excel model, the following lifeguards scheduled per day is the optimal solution:\
# Monday: 7 lifeguards\
# Tuesday: 5 lifeguards\
# Wednesday: 5 lifeguards\
# Thursday: 7 lifeguards\
# Friday: 8 lifeguards\
# Saturday: 10 lifeguards\
# Sunday: 10 lifeguards\
# Total Lifeguards (employees): 13

# ## Screenshot of Excel Solver - EmployeeScheduling
display.Image("lifeguard_solver.png")


# ## Python/Gurobi Model - EmployeeScheduling
from gurobipy import *
m = Model('lifeguard')
 
# Sets
## Sets of days
D = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
 
# Parameters
## Minimum number of employees required each day
r = [7,5,5,7,7,10,10]
## Workday adjacency matrix
A = [[1,0,0,0,1,1,1],\
    [1,1,0,0,0,1,1],\
    [1,1,1,0,0,0,1],\
    [1,1,1,1,0,0,0],\
    [0,1,1,1,1,0,0],\
    [0,0,1,1,1,1,0],\
    [0,0,0,1,1,1,1]]

# Decision Variables
## Number of employees, non-negativity included with lower-bound, using vtype to produce integer values in DVs
x = [m.addVar(name=D[j],lb=0) for j in range(len(D))]
m.update()
 
# Objective Function
## Minimize number of employees
m.setObjective(sum(x), GRB.MINIMIZE)
m.update()
 
# Constraints
## Number of employees constraint
m.addConstrs(quicksum(A[i][j]*x[j] for j in range(len(D))) >= r[i] for i in range(len(D)))
m.update()

# Solve and Print Solution
m.optimize()
print('\n\n')
for var in m.getVars():
    print('The company shoud schedule %s employees on %s.' % (round(var.x,2),var.varName))
print('The total number of employees is %s.' % round(m.objval,2))

###############################################################################################################################


# ## 2) Model Formulation - Multiperiod Planning
# 
# **Sets** \
# $W$: set of weeks \{0=Week_0, 1=Week_1, ..., 6=Week_5\}\
# $\hat{W}$: set of weeks including Week 0 \{0,1,2,3,4,5,6\}
# 
# **Parameters** \
# $D$: vector of demands (105,170,230,180,150,250) \
# $C$: per unit production costs (190,190,190,190,190,190) \
# $V$: per unit overtime production costs (260,260,260,260,260,260) \
# per unit holding costs = 10 dollars \
# initial inventory = 0 \
# production capacity = 160 \
# overtime production capacity = 50 
# 
# **Decision Variables** \
# $P$: vector amount of units to produce each week $W$ \
# $T$: vector amount of units to overtime produce each week $W$ \
# $I$: vector amount of inventory at the end of each week $\hat{W}$
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as 
# 
# \begin{equation*}
# \begin{matrix}
# \underset{P,T,I}{\min} & \underset{j \in W}{\sum} C_jP_j + V_jT_j + 10I_j  &\\
# \textrm{s.t.} & I_{j-1}+P_j + T_j & = & D_j+I_j, & \forall j\in W\\ 
# & I_0 & = & 0 &  \\
# & P_j & \leq & 160,  & \forall j\in W \\
# & T_j & \leq & 50, & \forall j\in W \\
# & P_j & \geq & 0, & \forall j\in W \\
# & T_j & \geq & 0, & \forall j\in W \\
# & I_j & \geq & 0, & \forall j\in \hat{W} \\
# \end{matrix}
# \end{equation*}

# ## Screenshot of Excel Model - Multiperiod Planning
display.Image("multiperiod_model.png")


# The factory should produce **105.0** snowboards, overtime produce **0.0** snowboards, and hold **0.0** in inventory in **Week 0**. \
# The factory should produce **160.0** snowboards, overtime produce **30.0** snowboards, and hold **20.0** in inventory in **Week 1**. \
# The factory should produce **160.0** snowboards, overtime produce **50.0** snowboards, and hold **0.0** in inventory in **Week 2**. \
# The factory should produce **160.0** snowboards, overtime produce **20.0** snowboards, and hold **0.0** in inventory in **Week 3**. \
# The factory should produce **160.0** snowboards, overtime produce **30.0** snowboards, and hold **40.0** in inventory in **Week 4**. \
# The factory should produce **160.0** snowboards, overtime produce **50.0** snowboards, and hold **0.0** in inventory in **Week 5**. \
# 
# The total cost is **$219,350**.

# ## Screenshot of Excel Solver - Multiperiod Planning
display.Image("multiperiod_solver.png")

# ## Python/Gurobi Model - Multiperiod Planning
from gurobipy import *
import numpy as np
m = Model('multiperiod')

# Sets
## Set of weeks
W = ['Week 0', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
Wh = ['Week 0', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']

# Parameters
## Demands
D = [105,170,230,180,150,250]
## Unit costs
C = [190,190,190,190,190,190]
V = [260,260,260,260,260,260]

# Decision Variables
## Number to produce each week
P = [m.addVar(name=W[j], lb=0, ub = 160) for j in range(len(W))]
## Number to overtime produce each week 
T = [m.addVar(name=W[j], lb=0, ub = 50) for j in range(len(W))]
## Inventory at the end of each week
I = [m.addVar(name=Wh[j], lb=0) for j in range(len(Wh))]
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(C[j]*P[j] + V[j]*T[j]  + 10*I[j] for j in range(len(W))), GRB.MINIMIZE)
m.update()

# Constraints
## Demand and inventory balance constraints
m.addConstrs(I[j-1] + P[j] + T[j] == D[j] + I[j] for j in range(len(W)))
## Initial inventory
m.addConstr(I[0] == 0)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for j in range(len(W)):
    print("The factory should produce %s snowboards, overtime produce %s snowboards, and hold %s in inventory in %s." % ( P[j].x, T[j].x, I[j].x, W[j] ))
print("\nThe total cost is %s." % "${:,.0f}".format(m.objval))

###############################################################################################################################


# ## 3) Model Formulation - Investment Planning
# 
# **Set** \
# $I$: set of investments \{0=Savings Account, 1=Cert. of Deposit, 2=Atlantic Lighting, 3=Arkansas REIT, 4=Bedrock Insurance,\
#                          5=Nocal Mining Bond, 6=Minicomp Systems, 7=Anthony Hotels}
# 
# **Parameters** \
# $E$: vector of expected returns (.04, .052, .071, .10, .082, .065, .20, .125) \
# $R$: vector of investment ratings (A, A, B+, B, A, B+, A, C) \
# $L$: vector of liquidities (immediate, 5-year, immediate, immediate, 1-year, 1-year, immediate, immediate) \
# $F$: vector of risk factors (0, 0, 25, 30, 20, 15, 65, 40)\
# Amount to invest: 100,000 dollars
# 
# 
# **Decision Variables** \
# $x_I$: amount to invest in each investment $I$ 
# 
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as 
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\min} & \underset{j \in I}{\sum} x_jF_j  &\\
# \textrm{s.t.} & \underset{j \in I}{\sum} E_jx_j & \geq & .075 * 100000 & \forall j\in I\\
# & x_0 + x_1 + x_4 + x_6 & \geq & .50 * 100000 & \\
# & x_0 + x_2 + x_3 + x_6 + x_7 & \geq & .40 * 100000 \\
# & x_0 + x_1 & \leq & .30 * 100000 \\
# & \underset{j \in I}{\sum} x_j & = & 100000 & \forall j\in I\\
# & x & \geq & 0
# \end{matrix}
# \end{equation*}

# ## Screenshot of Excel Model - Security Financial Planning
display.Image("security_model.png")


# The following investments should be made to meet the portfolio goals: \
# Savings Account: 17,333.33 dollars \
# Cert. of Deposit: 12,666.67 dollars \
# Atlantic Lighting: 0.00 dollars \
# Arkansas REIT: 22,66.67 dollars \
# Bedrock Insurance: 47,333.33 dollars \
# Nocal Mining Bond: 0.00 dollars \
# Minicomp Systems: 0.00 dollars \
# Antony Hotels: 0.00 dollars \ 
# 
# Total Risk is 1,626,666.67 points

# ## Screenshot of Excel Solver - Security Financial Planning
display.Image("security_solver.png")


# ## Python/Gurobi Model - Security Financial Planning
from gurobipy import *
m = Model('security financial planning')

# Sets
## Set of investments
I = ['Savings Account', 'Cert. of Deposit', 'Atlantic Lighting', 'Arkansas REIT', 'Bedrock Insurance',\
     'Nocal Mining Bond', 'Minicomp Systems', 'Anthony Hotels']

# Set Lengths
## Create ranges to simplify code
rI = range(len(I))

# Parameters
## Expected retuns
E = [.04,.052,.071,.10,.082,.065,.20,.125]
## Ratings
R = ['A', 'A', 'B+', 'B', 'A', 'B+', 'A', 'C']
## Liquidities
L = ['Immediate', '5-year', 'Immediate', 'Immediate', '1-year', '1-year', 'immediate', 'immediate']
## Risk Factors
F = [0,0,25,30,20,15,65,40]

# Decision Variables
x = [m.addVar(name=str(I[j]), lb=0) for j in  rI]
m.update()

# Objective Function
## Minimize risk
m.setObjective(quicksum(F[j]*x[j] for j in range(len(I))), GRB.MINIMIZE)

# Constraints
## Total Investment
m.addConstr(quicksum(x[j] for j in rI) == 100000)
## Expected Return
m.addConstr(quicksum(E[j]*x[j] for j in rI) >= 7500)
## A-Rated Investments
m.addConstr(x[0] + x[1] + x[4] + x[6] >= 50000)
## Liquidity 
m.addConstr(x[0] + x[2] + x[3] + x[6] + x[7] >= 40000)
## Savings and CDs
m.addConstr(x[0] + x[1] <= 30000)
m.update()

# Solve and Print Solution
m.optimize()
print('\n\n')
print("The optimal investment plan is:")
for var in m.getVars():
    print("%s = $%s" % (var.varName,round(var.x,3)))
print("\nThe total risk is %s points." % "{:,.0f}".format(m.objval))


########## NETWORK OPTIMIZATION ##########

# ## Problem - Shipping Vehicles from Plants to Regions of the Country

# ## Model Formulation
# 
# **Sets** \
# $R$: set of regions \{Region 1, Region 2, Region 3, Region 4\} \
# $P$: set of plants \{Plant 1, Plant 2, Plant 3\} \
# $A$: set of plant-region arcs
# 
# **Parameters** \
# $d$: vector of demands ($d_{Region 1} = 450$, $\ldots$, $d_{Region 4} = 300$) \
# $b$: vector of capacities ($b_{Plant 1} = 450$, $b_{Plant 2} = 600$, $b_{Plant 3} = 500$) \
# $C$: matrix of costs, where $c_{i,j}$ represents the shipping cost of an automobile from plant $i$ to region $j$
# 
# **Decision Variables** \
# $x_{i,j}$: the number of cars sent from plant $i$ to region $j$ 
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as
# 
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\min} & \underset{(i,j)\in A}{\sum}c_{i,j}x_{i,j} &\\
# \textrm{s.t.} & \underset{j:(i,j) \in A}{\sum}x_{i,j} & \leq & b_i & \forall i \in P \\ 
#  & \underset{i:(i,j) \in A}{\sum}x_{i,j} & \geq & d_i & \forall j \in R \\
# & x_{i,j} & \geq & 0 & \forall (i,j) \in A \\ 
# \end{matrix}
# \end{equation*}

# ## Transportation Model
from gurobipy import *
m = Model('Prob1')

# Sets and Parameters
## Set of regions, demand data
R, d = multidict({
    'Region 1': 450,
    'Region 2': 200,
    'Region 3': 300,
    'Region 4': 300})
## Set of plants, capacity data
P, b = multidict({
    'Plant 1': 450,
    'Plant 2': 600,
    'Plant 3': 500})
## Set of plant-region arcs, cost data
A, c = multidict({
    ('Plant 1','Region 1'): 131,
    ('Plant 1','Region 2'): 218,
    ('Plant 1','Region 3'): 266,
    ('Plant 1','Region 4'): 120,
    ('Plant 2','Region 1'): 250,
    ('Plant 2','Region 2'): 116,
    ('Plant 2','Region 3'): 263,
    ('Plant 2','Region 4'): 278,
    ('Plant 3','Region 1'): 178,
    ('Plant 3','Region 2'): 132,
    ('Plant 3','Region 3'): 122,
    ('Plant 3','Region 4'): 180})

# Decision Variables
## Number of cars shipped from plant i to region j
x = m.addVars(A, name='arc', lb=0)
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(c[i,j]* x[i,j] for (i,j) in A), GRB.MINIMIZE)
m.update()

# Constraints
## Capacity constraints
m.addConstrs(x.sum(i, '*') <= b[i] for i in P) # summing over the j indices and i is fixed
## Demand constraints
m.addConstrs(x.sum('*', j) >= d[j] for j in R)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for (i,j) in A:
    print("Ship %s vehicles from %s to %s." % (round(x[i,j].x),i,j))
print("\nThe total cost is %s." % "${:,.0f}".format(m.objval))

# Solution: 
# Ship 150 vehicles from Plant 1 to Region 1.
# Ship 0 vehicles from Plant 1 to Region 2.
# Ship 0 vehicles from Plant 1 to Region 3.
# Ship 300 vehicles from Plant 1 to Region 4.
# Ship 100 vehicles from Plant 2 to Region 1.
# Ship 200 vehicles from Plant 2 to Region 2.
# Ship 0 vehicles from Plant 2 to Region 3.
# Ship 0 vehicles from Plant 2 to Region 4.
# Ship 200 vehicles from Plant 3 to Region 1.
# Ship 0 vehicles from Plant 3 to Region 2.
# Ship 300 vehicles from Plant 3 to Region 3.
# Ship 0 vehicles from Plant 3 to Region 4.

# The total cost is $176,050.


# ## Problem - Assigning Buses to Routes

# ## Model Formulation
# 
# **Sets** \
# $P$: set of companies \{Company 1, Company 2, ..., Company 6\} \
# $R$: set of routes \{Route 1, Route 2, ..., Route 8\} \
# $A$: set of company-route assignment arcs
# 
# **Parameters** \
# $C$: matrix of costs, where $c_{i,j}$ represents the bid amount from company $i$ for route $j$ \
# A company can be assigned to at most two routes \
# Exactly one company must be assigned to each route
# 
# **Decision Variables** \
# $x_{i,j}$: =1 if company $i$ is assigned route $j$, =0 otherwise
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as
# 
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\min} & \underset{(i,j) \in A}{\sum}c_{i,j}x_{i,j} &\\
# \textrm{s.t.} & \underset{j:(i,j) \in A}{\sum}x_{i,j} & \leq & 2 & \forall i \in P \\ 
# & \underset{i:(i,j) \in A}{\sum}x_{i,j} & = & 1 & \forall j \in R \\ 
# & x_{i,j} & \geq & 0 & \forall (i,j) \in A \\ 
# \end{matrix}
# \end{equation*}


# company assigned is the i index
# in company constraint - only summing over the j values that are in the set of arcs
# one company to each route constraint - 

# ## Assignment Model
from gurobipy import *
m = Model('ex53')

# Sets and Parameters
## Set of companies
P = {'Company 1','Company 2','Company 3','Company 4','Company 5','Company 6'}
## Set of routes
R = {'Route 1','Route 2','Route 3','Route 4','Route 5','Route 6','Route 7','Route 8'}


## Set of plant-region arcs, cost data
A, c = multidict({
    ('Company 1','Route 2'): 8200,
    ('Company 1','Route 3'): 7800,
    ('Company 1','Route 4'): 5400,
    ('Company 1','Route 6'): 3900,
    ('Company 2','Route 1'): 7800,
    ('Company 2','Route 2'): 8200,
    ('Company 2','Route 4'): 6300,
    ('Company 2','Route 6'): 3300,
    ('Company 2','Route 7'): 4900,
    ('Company 3','Route 2'): 4800,
    ('Company 3','Route 6'): 4400,
    ('Company 3','Route 7'): 5600,
    ('Company 3','Route 8'): 3600,
    ('Company 4','Route 3'): 8000,
    ('Company 4','Route 4'): 5000,
    ('Company 4','Route 5'): 6800,
    ('Company 4','Route 7'): 6700,
    ('Company 4','Route 8'): 4200,
    ('Company 5','Route 1'): 7200,
    ('Company 5','Route 2'): 6400,
    ('Company 5','Route 4'): 3900,
    ('Company 5','Route 5'): 6400,
    ('Company 5','Route 6'): 2800,
    ('Company 5','Route 8'): 3000,
    ('Company 6','Route 1'): 7000,
    ('Company 6','Route 2'): 5800,
    ('Company 6','Route 3'): 7500,
    ('Company 6','Route 4'): 4500,
    ('Company 6','Route 5'): 5600,
    ('Company 6','Route 7'): 6000,
    ('Company 6','Route 8'): 4200})

# Decision Variables
## Assignments
x = m.addVars(A, name='arc', lb=0)
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in A), GRB.MINIMIZE)
m.update()

# Constraints
## A company can be assigned to at most two routes
# i index is in control of the loop and the j index is the one we want to sum over
m.addConstrs(x.sum(i, '*') <= 2 for i in P)
## Exactly one company must be assigned to each route
# j index is in control of the loop and the i index is the one we want to sum over
m.addConstrs(x.sum('*', j) == 1 for j in R)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for i,j in A:
    if x[i,j].x == 1:
        print("Assign %s to %s." % (i,j))
print("\nThe total cost is %s." % "${:,.0f}".format(m.objval))
# Solution:
# Assign Company 1 to Route 3.
# Assign Company 2 to Route 6.
# Assign Company 2 to Route 7.
# Assign Company 3 to Route 2.
# Assign Company 5 to Route 4.
# Assign Company 5 to Route 8.
# Assign Company 6 to Route 1.
# Assign Company 6 to Route 5.

# The total cost is $40,300.



# ## Problem - Shortest Walk Across the Country

# ## Model Formulation
# 
# **Sets** \
# $N$: set of cities {1,2,...,10}\
# $A$: set of city pair arcs
# 
# **Parameters** \
# $s$: vector of supply at each node \
# $d$: vector of demand at each node \
# $C$: matrix of distances, where $c_{i,j}$ is the distance from city $i$ to city $j$
# 
# **Decision Variables** \
# $x_{i,j}$: = 1 if the path includes the arc $(i,j)$, = 0 otherwise
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as
# 
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\min} & \underset{(i,j) \in A}{\sum}c_{i,j}x_{i,j} &\\
# \textrm{s.t.} & \underset{i: (i,j) \in A}{\sum}x_{i,j} + s_j & = & \underset{k: (j,k) \in A}{\sum}x_{j,k} + d_j & \forall j \in N \\
# & x_{i,j} & \leq & 0 & \forall (i,j) \in A \\
# \end{matrix}
# \end{equation*}


# initial node is the only node that has supply, all other nodes have 0 supply
# demand is such that the destination is the only node that has demand and is set to one, all others to 0

# i is where the flow is coming from on LHS
# j is where the flow is going to on RHS


# ## Shortest Path Model
from gurobipy import *
import numpy as np
m = Model('ex55')

# Sets and Parameters
## Set of cities, supply data, demand data
N, s, d = multidict({
    1: [1,0],
    2: [0,0],
    3: [0,0],
    4: [0,0],
    5: [0,0],
    6: [0,0],
    7: [0,0],
    8: [0,0],
    9: [0,0],
    10: [0,1]})
## Set of city-city arcs, distance data
A, c = multidict({
    (1,2): 70,
    (1,3): 63,
    (1,4): 56,
    (2,3): 25,
    (2,4): 19,
    (2,5): 73,
    (2,6): 50,
    (2,7): 79,
    (3,2): 25,
    (3,4): 29,
    (3,5): 69,
    (3,6): 61,
    (4,2): 19,
    (4,3): 29,
    (4,5): 67,
    (4,6): 45,
    (4,9): 85,
    (5,6): 18,
    (5,7): 67,
    (5,8): 69,
    (5,9): 54,
    (5,10): 87,
    (6,5): 18,
    (6,7): 72,
    (6,8): 52,
    (6,9): 51,
    (6,10): 97,
    (7,8): 17,
    (7,9): 31,
    (7,10): 72,
    (8,7): 17,
    (8,9): 15,
    (9,7): 31,
    (9,8): 15,
    (9,10): 69})


# Decision Variables
## Assignments (non-negativity caputred w/ lb)
x = m.addVars(A, name='arc', lb=0)
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in A), GRB.MINIMIZE)
m.update()

# Constraints
## Flow balance, plural 'constraints' bc we are adding one for each node
# LHS is sum up over all values coming in for node j
# RHS is sum up over all values going out for node j
m.addConstrs(x.sum('*',j) + s[j] == x.sum(j, '*') + d[j] for j in N)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for i,j in A:
    if x[i,j].x == 1:
        print("Travel from city %s to city %s." % (i,j))
print("\nThe total distance is %s." % m.objval)

# Solution:

# Travel from city 1 to city 4.
# Travel from city 4 to city 6.
# Travel from city 6 to city 10.

# The total distance is 198.0.


#Title: Network Optimization
# Show the supply chain model
from IPython import display
display.Image("GE_supply_chain_model.png")


# ## 3) Greedy Algorithm

# ## Algorithm Steps
# - Initialize
#     - Adjacency set
#     - path x
#     - unvisited list u
# - while x is incomplete:
#     - for unvisited adjacent nodes do: \
#             - add closest node to x \
#             - remove closest node from u

# Dictionary key values:
# 1: Boston
# 2: Chicago
# 3: Indianapolis
# 4: Memphis
# 5: Denver
# 6: Phoenix
# 7: Salt Lake City
# 8: Oakland
# 9: Seattle
# 10: Vancouver
# 11: Nome

# ## Greedy Heuristic for Shortest Path - GE Turbine Problem


# Node dictionary with adjacent nodes and times (in hours) with string values as the city
A = {
    'Boston': [['Chicago',63.66],['Indianapolis',62.98],['Memphis',81.30]],
    'Chicago': [['Denver',40.04],['Phoenix',75.06],['Salt Lake City',57.94]],
    'Indianapolis': [['Denver',41.66],['Phoenix',73.98],['Salt Lake City',60.82]],
    'Memphis': [['Denver',41.88],['Phoenix',58.48],['Salt Lake City',60.46]],
    'Denver': [['Oakland',54.98],['Seattle',57.42],['Vancouver',60.14]],
    'Phoenix': [['Oakland',34.82],['Seattle',58.16],['Vancouver',60.86]],
    'Salt Lake City': [['Oakland',34.60],['Seattle',36.80],['Vancouver',39.52]],
    'Oakland': [['Nome',108]],
    'Seattle': [['Nome',102]],
    'Vancouver': [['Nome',104]]
}

# Initialize parameters
## source
s = 'Boston'
## desintation
d = 'Nome'
## large number, larger than the sum of all times
M = 100000
## path
x = [s]
## unvisited nodes
u = ['Chicago','Indianapolis','Memphis','Denver','Phoenix','Salt Lake City','Oakland','Seattle','Vancouver',d]
## total time
total_time = 0


# loop until path reaches destination
while x[-1] != d:
    # reset closest time to a large number
    time = M
    # loop over nodes that are adjacent to the last node in the path
    for j in A[x[-1]]:
        # only consider unvisited nodes - referring to the index 0 position for the node
        if (j[0] in u):
            # choose the node that is closest
            if j[1] < time:
                # save its label
                k = j[0]
                # save its time
                time = j[1]
    # add closest node to the path
    x.append(k)
    # ...and remove it from the unvisited nodes list - working through the supply chain
    u.remove(k)
    # increment the total time
    total_time += time
    
# print the solution
print("The shortest path is: %s" % x)
print("The total travel time is %s hours." % total_time)


# ## What is the path and what is the total travel time in hours?

# Answer: The shortest path is: Boston to Indianapolis to Denver to Oakland to Nome. \
# The total travel time is 267.62 hours.

# ## 4) Model Formulation

# ## Model Formulation
# 
# **Sets** \
# $N$: set of cities {1,2,...,11}\
# $A$: set of city pair arcs
# 
# **Parameters** \
# $s$: vector of supply at each node \
# $d$: vector of demand at each node \
# $C$: matrix of times, where $c_{i,j}$ is the travel time from city $i$ to city $j$\
# one 10-hour rest must occur every 11 hours of driving
# 
# **Decision Variables** \
# $x_{i,j}$: = 1 if the path includes the arc $(i,j)$, = 0 otherwise
# 
# **Objective Function and Constraints** \
# The optimization model is formulated as
# 
# 
# \begin{equation*}
# \begin{matrix}
# \underset{x}{\min} & \underset{(i,j) \in A}{\sum}c_{i,j}x_{i,j} &\\
# \textrm{s.t.} & \underset{i: (i,j) \in A}{\sum}x_{i,j} + s_j & = & \underset{k: (j,k) \in A}{\sum}x_{j,k} + d_j & \forall j \in N \\
# & x_{i,j} & \leq & 0 & \forall (i,j) \in A \\
# \end{matrix}
# \end{equation*}


# initial node is the only node that has supply, all other nodes have 0 supply
# demand is such that the destination is the only node that has demand and is set to one, all others to 0

# i is where the flow is coming from on LHS
# j is where the flow is going to on RHS


# ## 5) Optimization Model

# trying to replace number values for nodes to city names
from gurobipy import *
import numpy as np
m = Model('GE_turbine_2')

# Sets and Parameters
## Set of cities, supply data, demand data
N, s, d = multidict({
    'Boston': [1,0],
    'Chicago': [0,0],
    'Indianapolis': [0,0],
    'Memphis': [0,0],
    'Denver': [0,0],
    'Phoenix': [0,0],
    'Salt Lake City': [0,0],
    'Oakland': [0,0],
    'Seattle': [0,0],
    'Vancouver': [0,0],
    'Nome': [0,1]})
## Set of city-city arcs, time data
A, c = multidict({
    ('Boston','Chicago'): 63.66,
    ('Boston','Indianapolis'): 62.98,
    ('Boston','Memphis'): 81.30,
    ('Chicago','Denver'): 40.04,
    ('Chicago','Phoenix'): 75.06,
    ('Chicago','Salt Lake City'): 57.94,
    ('Indianapolis','Denver'): 41.66,
    ('Indianapolis','Phoenix'): 73.98,
    ('Indianapolis','Salt Lake City'): 60.82,
    ('Memphis','Denver'): 41.88,
    ('Memphis','Phoenix'): 58.48,
    ('Memphis','Salt Lake City'): 60.46,
    ('Denver','Oakland'): 54.98,
    ('Denver','Seattle'): 57.42,
    ('Denver','Vancouver'): 60.14,
    ('Phoenix','Oakland'): 34.82,
    ('Phoenix','Seattle'): 58.16,
    ('Phoenix','Vancouver'): 60.86,
    ('Salt Lake City','Oakland'): 34.60,
    ('Salt Lake City','Seattle'): 36.80,
    ('Salt Lake City','Vancouver'): 39.52,
    ('Oakland','Nome'): 108,
    ('Seattle','Nome'): 102,
    ('Vancouver','Nome'): 104})


# Decision Variables
## Assignments (non-negativity caputred w/ lb)
x = m.addVars(A, name='arc', lb=0)
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in A), GRB.MINIMIZE)
m.update()

# Constraints
## Flow balance, plural 'constraints' bc we are adding one for each node
# LHS is sum up over all values coming in for node j
# RHS is sum up over all values going out for node j
m.addConstrs(x.sum('*',j) + s[j] == x.sum(j, '*') + d[j] for j in N)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for i,j in A:
    if x[i,j].x == 1:
        print("Travel from %s to %s." % (i,j))
print("\nThe total travel time is %s hours." % m.objval)


# **Answer: *The path is Boston to Chicago to Salt Lake City to Seattle to Nome. The total travel time is 260.4 hours.***

# ## 6) Business Interpretation and Sensitivity Analysis

# **Question: How should GE transport the turbines to Nome?**
# 
# Answer: GE should transport by trucks from Boston to Indianapolis to Denver to Oakland. From Oakland, GE should use boats to travel to Nome if based on the Greedy Heuristic Shortest Path Model.
# 
# GE should transport by trucks from Boston to Chicago to Salt Lake City to Seattle. From Seattle, GE should use boats to travel to Nome if based on the Optimization Model.

# **Question: How long should they expect it to take?**
# 
# Answer: GE should expect the travel to take 267.62 hours from Boston to Nome if using the Greedy Heuristic Shortest Path Model. \
# GE should expect the travel to take 260.4 hours from Boston to Nome if using the Optimization Model. \
# 
# In my experience, if I only had the option to choose from the two models in this scenario, I would select the model with the shortest time (optimization model). However, I would also consider whether using the optimization model for the path selection, but adding the slower time from the greedy heuristic to be more realistic when presenting to leadership. For example, explaining that the optimal path from the model is Boston to Chicago to Salt Lake City to Seattle to Nome and providing a longer time of 267.62 hours rather than 260.4 hours. I would consider this option to build in unexpected delays such as traffic, maintenance problems with the trucks, maintenance issues with material handling equipment at Boston or Nome, or even weather impacts. I would want to avoid presenting a hasty model and delivering my findings to drive a decision where there was not enough analysis conducted on potential schedule impacts. 

# **Question: How would you assess GE’s risk of incurring late penalties? (Hint: Do a sensitivity analysis to challenge one or
# more parameter assumptions.)**
# 
# Answer: Since GE is given two weeks to transport the turbines from Boston to Nome (2 weeks = 24 hours * 7 days per week * 2 weeks = 336 hours) and the results from the greedy heuristic is 267.62 hours and results from the optimization model are 260.40 hours, there is approximately 68 hours to spare from the greedy heuristic and 75 hours from the optimization model. Meeting for an average of 71.5 hours of additional time before late penalties are applied. 
# 
# However, these models assume that for each available hour of drive time that the drivers will achieve 50 miles per hour for all available hours of driving. Adding a level of realism that likely the driver will experience traffic, unexpected delays (maintenance) or other obstacles during their journey, I would almost immeidately reduce the expected drive speed by at least 5-10% for each arc. Considering this reduction of 7.5% (50 mph x .075 = 3.75, 50-3.75 = 46.25 miles per hour) the results are:
# 
# **a) Reduced Travel Speed Adjustment by 7.5%** \
# Greedy Heuristic Path: Boston to Indianapolis to Denver to Oakland to Nome (same)\
# Greedy Heuristic Travel Time (hours): 282.95 hours (+15.33 hours)\
# *Insights: From Indianapolis-Denver, an additional rest period was required (+10 hours) if the average travel speed decreases by 7.5% to 46.25 mph instead of 50 mph*.
# 
# Optimization Model Path: Boston to Chicago to Salt Lake City to Seattle to Nome.\
# Optimization Model Travel Time (hours): 265.62 hours (+5.22 hours).\
# *Insights: No additional rest periods were required in the Optimization Model.*
# 
# ----------------------------------------------------------------------------------------------------------------------------
# 
# **b) Added travel time with two traffic accidents**\
# Another consideration is whether a detour along the optimal path would change the route - such as two serious accidents along the journey between Boston to Chicago (+8 hours) and from Denver to Oakland (+10 hours). These accidents should impact the greedy heuristic and optimization model routes. This is building upon the reduced speed adjustment for more insights.\
# 
# Greedy Heuristic Path: The path is: Boston to Indianapolis to Denver to Seattle to Nome (change from Oakland to Seattle).\
# Greedy Heuristic Travel Time (hours): 279.58 hours (-3.37 hours from speed reduction by 7.5%).\
# *Insights: With an added accident along both the shortest path for the greedy heuristic and another on the optimization model, the total travel time **decreased** by 3.37 hours from the original assessment. The change from the shortest path going from Denver-Oakland to Denver-Seattle was a better option in this analysis.*
# 
# Optimization Model Path: Boston to Chicago to Salt Lake City to Seattle to Nome (same).\
# Optimization Model Travel Time (hours): 273.62 (+8 hours).\
# *Insights: The shortest path from the optimization model accepted the 8 hour additional traffic time from the accident and did not change the optimized path. 

# **Calculation work below for above sensitivity analysis in Question 6**

# a) Greedy Heuristic for Shortest Path Workspace for reduced travel speed by 7.5% adjustment for Question 6
# Node dictionary with adjacent nodes and times (in hours) with string values as the city instead of numbers
A = {
    'Boston': [['Chicago',65.25],['Indianapolis',64.52],['Memphis',83.51]],
    'Chicago': [['Denver',41.66],['Phoenix',77.90],['Salt Lake City',60.21]],
    'Indianapolis': [['Denver',53.42],['Phoenix',76.74],['Salt Lake City',73.32]],
    'Memphis': [['Denver',53.65],['Phoenix',60.79],['Salt Lake City',62.93]],
    'Denver': [['Oakland',57.01],['Seattle',59.64],['Vancouver',62.58]],
    'Phoenix': [['Oakland',36.02],['Seattle',60.44],['Vancouver',62.58]],
    'Salt Lake City': [['Oakland',35.78],['Seattle',38.16],['Vancouver',41.10]],
    'Oakland': [['Nome',108]],
    'Seattle': [['Nome',102]],
    'Vancouver': [['Nome',104]]
}

# Initialize parameters
## source
s = 'Boston'
## desintation
d = 'Nome'
## large number, larger than the sum of all times
M = 100000
## path
x = [s]
## unvisited nodes
u = ['Chicago','Indianapolis','Memphis','Denver','Phoenix','Salt Lake City','Oakland','Seattle','Vancouver',d]
## total time
total_time = 0


# loop until path reaches destination
while x[-1] != d:
    # reset closest time to a large number
    time = M
    # loop over nodes that are adjacent to the last node in the path
    for j in A[x[-1]]:
        # only consider unvisited nodes - referring to the index 0 position for the node
        if (j[0] in u):
            # choose the node that is closest
            if j[1] < time:
                # save its label
                k = j[0]
                # save its time
                time = j[1]
    # add closest node to the path
    x.append(k)
    # ...and remove it from the unvisited nodes list - working through the supply chain
    u.remove(k)
    # increment the total time
    total_time += time
    
# print the solution
print("The shortest path is: %s" % x)
print("The total travel time is %s hours." % total_time)


# a) Optimization Model Workspace for reduced travel speed by 7.5% adjustment for Question 6
from gurobipy import *
import numpy as np
m = Model('GE_turbine_2_slower')

# Sets and Parameters
## Set of cities, supply data, demand data
N, s, d = multidict({
    'Boston': [1,0],
    'Chicago': [0,0],
    'Indianapolis': [0,0],
    'Memphis': [0,0],
    'Denver': [0,0],
    'Phoenix': [0,0],
    'Salt Lake City': [0,0],
    'Oakland': [0,0],
    'Seattle': [0,0],
    'Vancouver': [0,0],
    'Nome': [0,1]})
## Set of city-city arcs, travel time data
A, c = multidict({
    ('Boston','Chicago'): 65.25,
    ('Boston','Indianapolis'): 64.52,
    ('Boston','Memphis'): 83.51,
    ('Chicago','Denver'): 41.66,
    ('Chicago','Phoenix'): 77.90,
    ('Chicago','Salt Lake City'): 60.21,
    ('Indianapolis','Denver'): 53.42,
    ('Indianapolis','Phoenix'): 76.74,
    ('Indianapolis','Salt Lake City'): 73.32,
    ('Memphis','Denver'): 53.65,
    ('Memphis','Phoenix'): 60.79,
    ('Memphis','Salt Lake City'): 62.93,
    ('Denver','Oakland'): 57.01,
    ('Denver','Seattle'): 59.64,
    ('Denver','Vancouver'): 62.58,
    ('Phoenix','Oakland'): 36.02,
    ('Phoenix','Seattle'): 60.44,
    ('Phoenix','Vancouver'): 62.58,
    ('Salt Lake City','Oakland'): 35.78,
    ('Salt Lake City','Seattle'): 38.16,
    ('Salt Lake City','Vancouver'): 41.10,
    ('Oakland','Nome'): 108,
    ('Seattle','Nome'): 102,
    ('Vancouver','Nome'): 104})


# Decision Variables
## Assignments (non-negativity caputred w/ lb)
x = m.addVars(A, name='arc', lb=0)
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in A), GRB.MINIMIZE)
m.update()

# Constraints
## Flow balance, plural 'constraints' bc we are adding one for each node
# LHS is sum up over all values coming in for node j
# RHS is sum up over all values going out for node j
m.addConstrs(x.sum('*',j) + s[j] == x.sum(j, '*') + d[j] for j in N)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for i,j in A:
    if x[i,j].x == 1:
        print("Travel from %s to %s." % (i,j))
print("\nThe total travel time is %s hours." % m.objval)


# b) Greedy Heuristic for Shortest Path Workspace for 7.5% slower travel and an accident between Boston to Chicago (+8 hours) 
# and Denver to Oakland (+10 hours) Question 6
# Node dictionary with adjacent nodes and times (in hours) with string values as the city instead of numbers
A = {
    'Boston': [['Chicago',73.25],['Indianapolis',64.52],['Memphis',83.51]],
    'Chicago': [['Denver',41.66],['Phoenix',77.90],['Salt Lake City',60.21]],
    'Indianapolis': [['Denver',53.42],['Phoenix',76.74],['Salt Lake City',73.32]],
    'Memphis': [['Denver',53.65],['Phoenix',60.79],['Salt Lake City',62.93]],
    'Denver': [['Oakland',67.01],['Seattle',59.64],['Vancouver',62.58]],
    'Phoenix': [['Oakland',36.02],['Seattle',60.44],['Vancouver',62.58]],
    'Salt Lake City': [['Oakland',35.78],['Seattle',38.16],['Vancouver',41.10]],
    'Oakland': [['Nome',108]],
    'Seattle': [['Nome',102]],
    'Vancouver': [['Nome',104]]
}

# Initialize parameters
## source
s = 'Boston'
## desintation
d = 'Nome'
## large number, larger than the sum of all times
M = 100000
## path
x = [s]
## unvisited nodes
u = ['Chicago','Indianapolis','Memphis','Denver','Phoenix','Salt Lake City','Oakland','Seattle','Vancouver',d]
## total time
total_time = 0


# loop until path reaches destination
while x[-1] != d:
    # reset closest time to a large number
    time = M
    # loop over nodes that are adjacent to the last node in the path
    for j in A[x[-1]]:
        # only consider unvisited nodes - referring to the index 0 position for the node
        if (j[0] in u):
            # choose the node that is closest
            if j[1] < time:
                # save its label
                k = j[0]
                # save its time
                time = j[1]
    # add closest node to the path
    x.append(k)
    # ...and remove it from the unvisited nodes list - working through the supply chain
    u.remove(k)
    # increment the total time
    total_time += time
    
# print the solution
print("The shortest path is: %s" % x)
print("The total travel time is %s hours." % total_time)


# In[9]:


# b) Optimization Model Workspace for for 7.5% slower travel and an accident between Boston to Chicago (+8 hours) 
# and Denver to Oakland (+10 hours) Question 6
from gurobipy import *
import numpy as np
m = Model('GE_turbine_2_slower_accidents')

# Sets and Parameters
## Set of cities, supply data, demand data
N, s, d = multidict({
    'Boston': [1,0],
    'Chicago': [0,0],
    'Indianapolis': [0,0],
    'Memphis': [0,0],
    'Denver': [0,0],
    'Phoenix': [0,0],
    'Salt Lake City': [0,0],
    'Oakland': [0,0],
    'Seattle': [0,0],
    'Vancouver': [0,0],
    'Nome': [0,1]})
## Set of city-city arcs, travel time data
A, c = multidict({
    ('Boston','Chicago'): 73.25,
    ('Boston','Indianapolis'): 64.52,
    ('Boston','Memphis'): 83.51,
    ('Chicago','Denver'): 41.66,
    ('Chicago','Phoenix'): 77.90,
    ('Chicago','Salt Lake City'): 60.21,
    ('Indianapolis','Denver'): 53.42,
    ('Indianapolis','Phoenix'): 76.74,
    ('Indianapolis','Salt Lake City'): 73.32,
    ('Memphis','Denver'): 53.65,
    ('Memphis','Phoenix'): 60.79,
    ('Memphis','Salt Lake City'): 62.93,
    ('Denver','Oakland'): 67.01,
    ('Denver','Seattle'): 59.64,
    ('Denver','Vancouver'): 62.58,
    ('Phoenix','Oakland'): 36.02,
    ('Phoenix','Seattle'): 60.44,
    ('Phoenix','Vancouver'): 62.58,
    ('Salt Lake City','Oakland'): 35.78,
    ('Salt Lake City','Seattle'): 38.16,
    ('Salt Lake City','Vancouver'): 41.10,
    ('Oakland','Nome'): 108,
    ('Seattle','Nome'): 102,
    ('Vancouver','Nome'): 104})


# Decision Variables
## Assignments (non-negativity caputred w/ lb)
x = m.addVars(A, name='arc', lb=0)
m.update()

# Objective Function
## Minimize total cost
m.setObjective(quicksum(c[i,j]*x[i,j] for (i,j) in A), GRB.MINIMIZE)
m.update()

# Constraints
## Flow balance, plural 'constraints' bc we are adding one for each node
# LHS is sum up over all values coming in for node i
# RHS is sum up over all values going out for node j
m.addConstrs(x.sum('*',j) + s[j] == x.sum(j, '*') + d[j] for j in N)
m.update()

# Solve and Print Solution
m.optimize()
print("\n\n")
for i,j in A:
    if x[i,j].x == 1:
        print("Travel from %s to %s." % (i,j))
print("\nThe total travel time is %s hours." % m.objval)


