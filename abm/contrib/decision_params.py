#### W #####
# Excitatory threshold above which a relocation is initiated
T_w = 0.5
# Social excitability
Eps_w = 3
# w decay time constant
g_w = 0.085
# Baseline of decision process
B_w = 0
# max value for w
w_max = 1

#### U #####
# Refractory threshold above which u resets decision w
T_u = 0.5
# Sensitivity of u to nearby agents
Eps_u = 3
# Timeconstant of u decay
g_u = 0.085
# Baseline of refractory variable u
B_u = 0
# max value for u
u_max = 1

##### Inhibition ####
S_wu = 0.25  # strength from w to u
S_uw = 0.01  # strength from u to w

##### Calculating Private Information #####
Tau = 10
F_N = 2
F_R = 1