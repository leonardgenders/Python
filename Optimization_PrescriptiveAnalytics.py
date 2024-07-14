# ## Title: M2 Linear Programming Assignment
# ## Author: Leo Genders
# ## Date: 14 July 2024

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
