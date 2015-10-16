
using Toms566


function newtmin( obj, x0; probnum=1, maxIts=100, relTol=1e-6, absTol=1e-2)
    # Minimize a function f using a Quasi-Newton's method.
    # obj: a function that evaluates the objective value,
    #      gradient, and Hessian at a point x, i.e.,
    #      (f, g, H) = obj(x, pnum)
    # x0:  starting point
    # probnum: Problem Number. Allows one object to contain
    #          multiple functions.
    # maxIts (optional): maximum number of iterations
    # relTol (optional): optimality tolerance based on
    #                    ||grad(x)|| <= optTol*||grad(x0)||
    # absTol (optional): optimality tolerance based on
    #                    ||grad(x)|| <= absTol
    
    # Initialize Process
    mu=1e-4
    its=0
    (f, g, H) = obj(x0,probnum)
    g0=g
    x=x0

    # Begin Minimization
    while ((norm(g) > (norm(g0)*relTol)) || (norm(g) > absTol)) && (its <= maxIts)     
        # Hessian Conditioning: 
        #   1) Make all negative eigenvalues positive
        #   2) Make all small eigenvales greater than a threshhold value, del
        #   3) Vary del so that it decreases as the gradient decreases
        # 
        # Note that I am modifying positive definite matrices as well as indefinite
        # matrices. Positive definite matrices may still have eigenvalues very close
        # to zero, so inverting that matrix can lead to very large terms. The method
        # I'm using places an upper bound on B^-1, but relaxes that bound as |g|->0.
        # 
        eps=1e-6
        del=min(eps,norm(g))
        (L,V)=eig(H)
        Lb=max(abs(L),del)
        Hb=V*diagm(Lb)*V'
             
        # Determine step direction
        d=-Hb\g
        
        # Backtracking linesearch
        alfa=1
        (fn, gn, Hn) = obj(x+alfa*d,probnum)
        while fn > f + alfa*mu*dot(g,d)
            alfa=alfa/2
            (fn, gn, Hn) = obj(x+alfa*d,probnum)
        end
        
        # Take a step
        x=x+alfa*d
        its=its+1        
        
        # Evaluate function for new x
        (f, g, H) = obj(x,probnum)
        
    end
    
    return(x, f, norm(g), its)
end

# Define objects to handle (f, g, H) = func(x)
function testobj(x,pnum)
    # Test problem: Least Squares (Pregenerated random A, b)
    A=[-0.2437    0.1873;
        0.2157   -0.0825;
       -1.1658   -1.9330;
       -1.1480   -0.4390;
        0.1049   -1.7947;
        0.7223    0.8404;
        2.5855   -0.8880;
       -0.6669    0.1001]
    b=[-0.5445;
        0.3035;
       -0.6003;
        0.4900;
        0.7394;
        1.7119;
       -0.1941;
       -2.1384]
    f=(1/2)*norm(A*x-b)^2
    g=A'*(A*x-b)
    H=A'A
    return(f,g,H)
end

function tomsobj(x, pnum)
    p=Problem(pnum)
    f=p.obj(x)
    g=p.grd(x)
    H=p.hes(x)
    n=p.n
    return(f,g,H)
end

function logregobj(x, pnum)
    # Logrithmic Regression: Admissions problem
    # Admissions data set. I know this is an ugly way to get the data in but I can't get readtable() to work in Juliabox.
    admit=[0;1;1;1;0;1;1;0;1;0;0;0;1;0;1;0;0;0;0;1;0;1;0;0;1;1;1;1;1;0;0;0;0;1;0;0;0;0;1;1;0;1;1;0;0;1;1;0;0;0;0;0;0;1;0;1;0;0;0;0;1;0;0;1;0;0;0;0;0;0;0;0;0;0;0;0;0;1;0;1;0;0;0;0;1;0;0;0;0;1;0;1;0;0;1;0;0;0;0;0;0;0;0;0;1;1;1;0;0;0;0;0;0;0;0;0;1;0;1;0;1;1;0;0;0;0;1;0;0;0;1;0;0;0;0;0;0;0;0;1;0;1;0;0;0;0;0;0;1;0;1;0;1;0;0;1;0;1;0;0;0;0;1;0;0;0;0;0;0;0;0;0;0;1;0;1;0;1;0;0;0;0;0;1;0;0;0;0;0;0;1;0;0;0;1;0;0;1;0;0;0;1;1;0;1;1;0;1;0;0;0;0;0;0;1;1;0;1;0;1;0;0;1;0;0;1;0;0;0;1;0;0;0;0;1;0;1;0;0;0;0;1;1;0;0;0;0;0;0;0;0;0;1;1;1;0;1;1;0;0;0;0;1;1;1;0;0;1;1;0;1;0;1;0;0;1;0;1;1;1;0;0;0;0;1;0;1;1;0;0;1;0;0;0;0;0;0;0;0;0;0;1;1;1;0;0;1;0;0;0;0;0;0;1;0;1;1;1;1;0;0;0;0;0;0;0;0;1;0;0;0;0;0;0;1;1;0;0;0;1;0;1;0;0;0;0;0;0;0;0;1;0;1;0;1;1;0;0;1;0;1;1;0;0;1;0;0;0;0;0;1;1;1;1;0;0;0;1;0;0;0;1;0;0;1;0;1;0;0;0;1;1;1;1;1;0;0;0;0;0]
    gre=[380;660;800;640;520;760;560;400;540;700;800;440;760;700;700;480;780;360;800;540;500;660;600;680;760;800;620;520;780;520;540;760;600;800;360;400;580;520;500;520;560;580;600;500;700;460;580;500;440;400;640;440;740;680;660;740;560;380;400;600;620;560;640;680;580;600;740;620;580;800;640;300;480;580;720;720;560;800;540;620;700;620;500;380;500;520;600;600;700;660;700;720;800;580;660;660;640;480;700;400;340;580;380;540;660;740;700;480;400;480;680;420;360;600;720;620;440;700;800;340;520;480;520;500;720;540;600;740;540;460;620;640;580;500;560;500;560;700;620;600;640;700;620;580;580;380;480;560;480;740;800;400;640;580;620;580;560;480;660;700;600;640;700;520;580;700;440;720;500;600;400;540;680;800;500;620;520;620;620;300;620;500;700;540;500;800;560;580;560;500;640;800;640;380;600;560;660;400;600;580;800;580;700;420;600;780;740;640;540;580;740;580;460;640;600;660;340;460;460;560;540;680;480;800;800;720;620;540;480;720;580;600;380;420;800;620;660;480;500;700;440;520;680;620;540;800;680;440;680;640;660;620;520;540;740;640;520;620;520;640;680;440;520;620;520;380;560;600;680;500;640;540;680;660;520;600;460;580;680;660;660;360;660;520;440;600;800;660;800;420;620;800;680;800;480;520;560;460;540;720;640;660;400;680;220;580;540;580;540;440;560;660;660;520;540;300;340;780;480;540;460;460;500;420;520;680;680;560;580;500;740;660;420;560;460;620;520;620;540;660;500;560;500;580;520;500;600;580;400;620;780;620;580;700;540;760;700;720;560;720;520;540;680;460;560;480;460;620;580;800;540;680;680;620;560;560;620;800;640;540;700;540;540;660;480;420;740;580;640;640;800;660;600;620;460;620;560;460;700;600]
    gpa=[3.61;3.67;4;3.19;2.93;3;2.98;3.08;3.39;3.92;4;3.22;4;3.08;4;3.44;3.87;2.56;3.75;3.81;3.17;3.63;2.82;3.19;3.35;3.66;3.61;3.74;3.22;3.29;3.78;3.35;3.4;4;3.14;3.05;3.25;2.9;3.13;2.68;2.42;3.32;3.15;3.31;2.94;3.45;3.46;2.97;2.48;3.35;3.86;3.13;3.37;3.27;3.34;4;3.19;2.94;3.65;2.82;3.18;3.32;3.67;3.85;4;3.59;3.62;3.3;3.69;3.73;4;2.92;3.39;4;3.45;4;3.36;4;3.12;4;2.9;3.07;2.71;2.91;3.6;2.98;3.32;3.48;3.28;4;3.83;3.64;3.9;2.93;3.44;3.33;3.52;3.57;2.88;3.31;3.15;3.57;3.33;3.94;3.95;2.97;3.56;3.13;2.93;3.45;3.08;3.41;3;3.22;3.84;3.99;3.45;3.72;3.7;2.92;3.74;2.67;2.85;2.98;3.88;3.38;3.54;3.74;3.19;3.15;3.17;2.79;3.4;3.08;2.95;3.57;3.33;4;3.4;3.58;3.93;3.52;3.94;3.4;3.4;3.43;3.4;2.71;2.91;3.31;3.74;3.38;3.94;3.46;3.69;2.86;2.52;3.58;3.49;3.82;3.13;3.5;3.56;2.73;3.3;4;3.24;3.77;4;3.62;3.51;2.81;3.48;3.43;3.53;3.37;2.62;3.23;3.33;3.01;3.78;3.88;4;3.84;2.79;3.6;3.61;2.88;3.07;3.35;2.94;3.54;3.76;3.59;3.47;3.59;3.07;3.23;3.63;3.77;3.31;3.2;4;3.92;3.89;3.8;3.54;3.63;3.16;3.5;3.34;3.02;2.87;3.38;3.56;2.91;2.9;3.64;2.98;3.59;3.28;3.99;3.02;3.47;2.9;3.5;3.58;3.02;3.43;3.42;3.29;3.28;3.38;2.67;3.53;3.05;3.49;4;2.86;3.45;2.76;3.81;2.96;3.22;3.04;3.91;3.34;3.17;3.64;3.73;3.31;3.21;4;3.55;3.52;3.35;3.3;3.95;3.51;3.81;3.11;3.15;3.19;3.95;3.9;3.34;3.24;3.64;3.46;2.81;3.95;3.33;3.67;3.32;3.12;2.98;3.77;3.58;3;3.14;3.94;3.27;3.45;3.1;3.39;3.31;3.22;3.7;3.15;2.26;3.45;2.78;3.7;3.97;2.55;3.25;3.16;3.07;3.5;3.4;3.3;3.6;3.15;3.98;2.83;3.46;3.17;3.51;3.13;2.98;4;3.67;3.77;3.65;3.46;2.84;3;3.63;3.71;3.28;3.14;3.58;3.01;2.69;2.7;3.9;3.31;3.48;3.34;2.93;4;3.59;2.96;3.43;3.64;3.71;3.15;3.09;3.2;3.47;3.23;2.65;3.95;3.06;3.35;3.03;3.35;3.8;3.36;2.85;4;3.43;3.12;3.52;3.78;2.81;3.27;3.31;3.69;3.94;4;3.49;3.14;3.44;3.36;2.78;2.93;3.63;4;3.89;3.77;3.76;2.42;3.37;3.78;3.49;3.63;4;3.12;2.7;3.65;3.49;3.51;4;2.62;3.02;3.86;3.36;3.17;3.51;3.05;3.88;3.38;3.75;3.99;4;3.04;2.63;3.65;3.89]
    rank=[3;3;1;4;4;2;1;2;3;2;4;1;1;2;1;3;4;3;2;1;3;2;4;4;2;1;1;4;2;1;4;3;3;3;1;2;1;3;2;3;2;2;2;3;2;3;2;4;4;3;3;4;4;2;3;3;3;3;2;4;2;4;3;3;3;2;4;1;1;1;3;4;4;2;4;3;3;3;1;1;4;2;2;4;3;2;2;2;1;2;2;1;2;2;2;2;4;2;2;3;3;3;4;3;2;2;1;2;3;2;4;4;3;1;3;3;2;2;1;3;2;2;3;3;3;4;1;4;2;4;2;2;2;3;2;3;4;3;2;1;2;4;4;3;4;3;2;3;1;1;1;2;2;3;3;4;2;1;2;3;2;2;2;2;2;1;4;3;3;3;3;3;3;2;4;2;2;3;3;3;3;4;2;2;4;2;3;2;2;2;2;3;3;4;2;2;3;4;3;4;3;2;1;4;1;3;1;1;3;2;4;2;2;3;2;3;1;1;1;2;3;3;1;3;2;3;2;4;2;2;4;3;2;3;1;2;2;2;4;3;2;1;3;2;1;3;2;2;3;3;4;4;2;4;4;3;2;3;2;2;2;2;3;3;3;3;4;3;2;3;2;3;2;1;2;2;3;1;4;2;2;3;4;4;2;4;1;4;4;4;2;2;2;1;1;3;1;2;2;3;2;3;2;2;3;4;1;2;2;3;3;2;3;4;4;2;2;4;4;1;3;2;4;2;3;1;2;2;2;4;3;3;1;3;3;1;3;4;1;3;4;3;4;2;3;3;2;2;2;2;2;3;3;2;2;1;2;1;3;3;1;1;2;2;1;3;3;3;1;2;2;3;1;1;2;4;2;2;3;2;2;2;2;1;2;1;2;2;2;2;2;2;3;2;3;2;3;2;2;3]
    
    y=admit;
    u=[ones(400,1) gre gpa rank]
    # x=[b a_gre a_gpa a_rank]

    # evaluate function
    f = -(sum((u*x).*(y.==1))-sum(log(1+exp(u*x))))

    # evaluate gradient
    g1 = -(sum(u[:,1].*(y.==1))-sum(u[:,1]./(1+exp(-u*x))))
    g2 = -(sum(u[:,2].*(y.==1))-sum(u[:,2]./(1+exp(-u*x))))
    g3 = -(sum(u[:,3].*(y.==1))-sum(u[:,3]./(1+exp(-u*x))))
    g4 = -(sum(u[:,4].*(y.==1))-sum(u[:,4]./(1+exp(-u*x))))

    g=[g1;g2;g3;g4]

    # evaluate hessian
    # I'm sure there's a more compact way to do this, but it works
    H11=sum(u[:,1].*u[:,1].*exp(u*x)./((1+exp(u*x)).^2))
    H12=sum(u[:,1].*u[:,2].*exp(u*x)./((1+exp(u*x)).^2))
    H13=sum(u[:,1].*u[:,3].*exp(u*x)./((1+exp(u*x)).^2))
    H14=sum(u[:,1].*u[:,4].*exp(u*x)./((1+exp(u*x)).^2))
    H22=sum(u[:,2].*u[:,2].*exp(u*x)./((1+exp(u*x)).^2))
    H23=sum(u[:,2].*u[:,3].*exp(u*x)./((1+exp(u*x)).^2))
    H24=sum(u[:,2].*u[:,4].*exp(u*x)./((1+exp(u*x)).^2))
    H33=sum(u[:,3].*u[:,3].*exp(u*x)./((1+exp(u*x)).^2))
    H34=sum(u[:,3].*u[:,4].*exp(u*x)./((1+exp(u*x)).^2))
    H44=sum(u[:,4].*u[:,4].*exp(u*x)./((1+exp(u*x)).^2))

    H=[H11 H12 H13 H14;H12 H22 H23 H24;H13 H23 H33 H34;H14 H24 H34 H44]
    
    #H=eye(4)
    
    return(f,g,H)
end

# Define problem to solve, call solver
pnum=18

p = Problem(pnum)
x0=p.x0
g0=p.grd(x0)

(xs, fs, grad,k)=newtmin(tomsobj,xs,probnum=pnum,maxIts=1000000,relTol=1e-3,absTol=1e-2)
gg0=grad/norm(g0);
println("f*=$fs")
println("g*=$grad")
println("g*./g0=$gg0")
println("its=$k")

# Results for Relative Tolerance = 1e-6, Absolute Tolerance = 1e-2 unless otherwise noted
#  No.  Name                             f(x*)      |∇f(x*)|    Iterations
# Test Least Squares                   4.19e+00     1.40e-15           1 Note: Starting at [0,0]
#   1  Hellical valley                 7.28e-10     1.69e-04          14
#   2  Bigg's EXP6                     3.96e-07     2.55e-06      604065
#   3  Gaussian                        1.13e-08     9.70e-11           2
#   4  Powell                          3.04e-14     1.39e-03          88
#   5  Box 3-dim                       2.06e-09     2.69e-05           9
#   6  Variably dimensioned            2.47e-01     1.00e+00        7068 Solved to absTol=1e0
#   7  Watson                          4.06e-05     1.77e-04       99163
#   8  Penalty I                       5.43e-04     8.47e-03          17 
#   9  Penalty II                      8.80e+01     9.72e-03         631
#  10  Brown badly scaled              3.22e-26     3.59e-07           8
#  11  Brown and Denis                 8.58e+04     8.58e-03          22
#  12  Gulf research and development   8.06e-12     1.60e-05          33
#  13  Trigonometric                   1.31e-09     5.20e-08        2387
#  14  Extended rosenbrock             1.27e-04     1.04e-02      997871 Solved to relTol=1e-5, absTol=1e-1
#  15  Extended Powell singular        8.40e-04     1.78e-02       16242 Solved to relTol=1e-5, absTol=1e-1
#  16  Beale                           6.17e-18     6.88e-09           7
#  17  Wood                            7.67e-11     9.76e-05          37
#  18  Chebyquad                       5.39e-03     1.47e-03     ~250000 Solved to relTol = 5.5e-4 
#  HW2 Logistic Regression             2.30e+02     1.30e-04           4 Note: Starting at [0,0,0,0]

# For comparison, gradient descent took ~150,000,000 steps to get to |∇f(x)|=0.0458 for the same x*. Granted, there was no
# linesearch, so it's not a perfect comparison, but obviously this is much better.

# There is definitely still room for improvement in my algorithm. It can get all problems to converge, but some of the more
# difficult problems take a pretty large number of iterations. I tried using some other methods to speed up convergence
# (see http://web.stanford.edu/class/cme304/docs/newton-type-methods.pdf), but I don't think I was implementing them right
# as they were pretty hit or miss in terms of performance. 

