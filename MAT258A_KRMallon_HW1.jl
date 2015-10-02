# Kevin Mallon
# UC Davis
# MAT 258a Numerical Optimization
# Homework 1 Problem 5

## Do not grade - see MAT258A_KRMallon_HW1.m

# ----------------------------
# Pkg.add("Convex")
# Pkg.add("SCS")
# Pkg.add("PyPlot")
# Pkg.add("ECOS")

using ECOS
using Convex
using PyPlot
# using Gadfly

# solver = SCSSolver(verbose=0)
# set_default_solver(solver);
# ----------------------------

A=randn(100,100)
b=randn(100)

x=Variable(100)

expr2norm=0.5*norm(A*x-b,2)^2+1*norm(x,1)

problem = minimize(expr2norm)
solve!(problem,ECOSSolver())

problem.optval
x.value

#println(round(problem.optval, 2))
#println(round(x.value, 2))

XHist=hist(x.value,100)

# plot(x=XHist[1],y=XHist[2],Geom.bar)
figure
#r=-0.25:0.1/50:0.25
plt[:hist](x.value,50,facecolor="w")
#plot(r,1.5*abs(r),"k")
