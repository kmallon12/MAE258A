# Kevin Mallon
# UC Davis
# MAT 258a Numerical Optimization
# Homework 1 Problem 5

# ----------------------------
# Pkg.add("Convex")
# Pkg.add("SCS")
# Pkg.add("PyPlot")

using SCS
using Convex
# using PyPlot
using Gadfly

solver = SCSSolver(verbose=0)
set_default_solver(solver);
# ----------------------------

A=randn(100,30)
b=randn(100)

x=Variable(30)

expr2norm=0.5*norm(A*x-b,2)^2

problem = minimize(expr2norm)
solve!(problem)

problem.optval
x.value

#println(round(problem.optval, 2))
#println(round(x.value, 2))

r=b-A*x.value

ResHist=hist(r,100)

plot(x=ResHist[1],y=ResHist[2],Geom.bar)
