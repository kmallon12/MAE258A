
using Toms566

function newtmin( obj, x0; pnum=1, maxIts=100, relTol=1e-6, absTol=1e-2)    
    
    # Initialize Process
    its=0
    (f, g, H) = obj(x0,pnum)
    g0=g
    x=x0

    # Begin Minimization
    while ((norm(g) > (norm(g0)*relTol)) || (norm(g) > absTol)) && (its <= maxIts)     
        eps=1e-6                                # Minimum eigenvalue value
        del=min(eps,norm(g))                    # Adjust eigenvalue minimum for small gradients
        (L,V)=eig(H)                            # Decompose Hessian
        Lb=max(abs(L),del)                      # Modify eigenvalues
        Hb=V*diagm(Lb)*V'                       # Rebuild Hessian                  
        d=-Hb\g                                 # Determine step direction
        alfa=armijo_search(x,d,obj,pnum)        # Backtracking linesearch
        x=x+alfa*d                              # Take a step
        its=its+1                               # Increase step counter
        (f, g, H) = obj(x,pnum)                 # Evaluate function for new x
    end
    
    return(x, f, norm(g), its)
end

function cgrdmin( obj, x0; pnum=1, maxIts=100, relTol=1e-6, absTol=1e-2)
  
    # Initialize Process
    its=0
    (f, g, H) = obj(x0,pnum)
    g0=g
    x=x0

    # Begin Minimization
    while ((norm(g) > (norm(g0)*relTol)) || (norm(g) > absTol)) && (its <= maxIts)     
        eps=1e-6                                # Minimum eigenvalue value
        del=min(eps,norm(g))                    # Adjust eigenvalue minimum for small gradients
        (L,V)=eig(H)                            # Decompose Hessian
        Lb=max(abs(L),del)                      # Modify eigenvalues
        Hb=V*diagm(Lb)*V'                       # Rebuild Hessian   
        d=conjgrad(Hb,g)                        # Determine step direction with CG algorithm
        alfa=armijo_search(x,d,obj,pnum)        # Backtracking linesearch
        x=x+alfa*d                              # Take a step
        its=its+1                               # Increase step counter
        (f, g, H) = obj(x,pnum)                 # Evaluate function for new x
    end
    
    return(x, f, norm(g), its)
end

function conjgrad(Q,b)
    # Looking for  d that minimizes 1/2 <d,Bd>+<g,d>, 
    # analogous to x that minimizes 1/2 <x,Qx>+<b,x>.
    # Q of the CG algorithm is H (Hessian) of the main function
    # b of the CG algorithm is g=∇f(x) of the main funtion
    n=length(b)     # Max iterations of CG
    x=0.*b          # Start CG at 0
    g=Q*x+b         # Initial gradient
    d=-g            # Initial direction
    k=1             # Iteration Counter
    r=g+Q*d         # Initialize residual
    while k<=n && (norm(r)/norm(g))>1e-3
        a = -dot(g,d)/dot(Q*d,d)    # Compute step length
        x =  x+a*d                  # Compute new x
        g =  g+a*Q*d                # Compute new gradient
        β =  dot(Q*d,g)/dot(Q*d,d)  # Compute beta
        d = -g+β*d                  # Compute new direction
        r =  g+Q*d                  # Compute residual
        k =  k+1                    # Increase step counter
    end
    return(x) 
end

function bfgsmin( obj, x0; pnum=1, maxIts=100, relTol=1e-6, absTol=1e-2)
    
    # Initialize Process
    its=0
    (f, g, H) = obj(x0,pnum)
    g0=g
    x=x0
    
    # Initialize with Gradient Descent
    Binv=eye(length(g))                     # Start with B=I
    d=-Binv*g                               # Determine step direction
    (alfa,d)=wolfe_search(x,d,obj,pnum,its) # Backtracking linesearch
    x=x+alfa*d                              # Take a step
    its=its+1                               # Increase step counter
    gold=g                                  # Save previous gradient
    (f, g, H) = obj(x,pnum)                 # Evaluate function for new x

    # Begin Minimization
    while ((norm(g) > (norm(g0)*relTol)) || (norm(g) > absTol)) && (its <= maxIts)
        s=alfa*d                            # Compute s
        y=g-gold                            # Compute y
        Binv=Binv+(dot(s,y)+dot(y,Binv*y))*(s*s')/dot(s,y)^2-(Binv*y*s'+s*y'*Binv)/dot(s,y) # Determine B^-1
        d=-Binv*g
        alfa=wolfe_search(x,d,obj,pnum)     # Backtracking linesearch
        xold=x
        x=x+alfa*d                          # Take a step
        its=its+1                           # Increase step counter
        gold=g                              # Save previous gradient
        (f, g, H) = obj(x,pnum)             # Evaluate function for new x 
    end
    
    return(x, f, norm(g), its)
end

# Define objects to handle (f, g, H) = func(x)
function tomsobj(x, pnum)
    p=Problem(pnum)
    f=p.obj(x)
    g=p.grd(x)
    H=p.hes(x)
    n=p.n
    return(f,g,H)
end

# Backtracking linesearch (Armijo)
function armijo_search(x,d,obj,pnum)
    mu=1e-4
    alfa=1
    (f0, g0) = obj(x,pnum)
    (fn, gn) = obj(x+alfa*d,pnum)
    while fn > f0 + alfa*mu*dot(g0,d)
        alfa=alfa/2
        (fn, gn) = obj(x+alfa*d,pnum)
    end
    return(alfa)
end

# Forward/backward linesearch (Wolfe)
function wolfe_search(x,d,obj,pnum) 
    # First, find an alfa that meets the Armijo condition. Then,
    # back up one step and search for an alfa between 0 and the
    # Armijo alfa that meets the Wolfe condition. 
    
    # Consider the Armijo condition alfa the maximum alfa and
    # zero as the minimum alfa. Evaluate the function at the
    # mean of the min and max alpha, and redefine the minimum
    # or maximum alfa based on the result. Repeat this process 
    # until either the Wolfe condition has been met or 25
    # iterations have been reached. 
    
    # The bound on the Wolfe conditions is relaxed with each
    # iteration. At the given maximum iterations, the min/max
    # alpha is less than a thousandth of the original range, so
    # at that point the goal is only to confirm if B will at 
    # least remain positive definite. 
    
    # Armijo condition linesearch
    mu=1e-4
    alfa_ar=1
    (f0, g0) = obj(x,pnum)
    (fn, gn) = obj(x+alfa_ar*d,pnum)
    k=1
    while fn > f0 + alfa_ar*mu*dot(g0,d)
        alfa_ar=0.5*alfa_ar
        (fn,~)=obj(x+alfa_ar*d,pnum)
    end 
    
    # Wolfe condition linesearch
    k=1
    maxits=25                                           # Maximum iterations
    eta=[logspace(-3,0,maxits);1]                       # W.C. bound: relax as k->maxits
    alfa_min=0                                          # Initial lower bound on alfa
    alfa_max=alfa_ar/0.5                                # Initial upper bound on alfa
    
    alfa=(alfa_min+alfa_max)/2                          # Initial guess for alfa
    alfa_lo=(alfa_min+alfa)/2
    alfa_hi=(alfa+alfa_max)/2
    
    (fmin,~) = obj(x+alfa_min*d,pnum)                   #                       
    (f,g) = obj(x+alfa*d,pnum)                          # Evaluate at the guess and at each bound.                    #
    (fmax,~) = obj(x+alfa_max*d,pnum)                   #
    
    while abs(dot(g,d)/dot(g0,d))>eta[k] && k<=maxits
        if f>=fmax && f>=fmin                           # If the function value is greater at the guess than at 
            alfa=(alfa_max-alfa_min)*rand(1)+alfa_min   # either bound, make a new guess.
        else
            if f < fmax && f <fmin                      #
                if fmin>fmax                            #
                    alfa_max=alfa_hi                    #
                else                                    #
                    alfa_min=alfa_lo                    # 
                end                                     # Redefine bounds based on function values
            elseif fmin < f && f < fmax                 #
                alfa_max=alfa_hi                        #
            elseif fmax < f && f < fmin                 #
                alfa_min=alfa_lo                        #
            end                                         
            alfa=(alfa_min+alfa_max)/2                  # Make a guess between new bounds.
        end
        alfa=(alfa_min+alfa_max)/2                      # Initial guess for alfa
        alfa_lo=(alfa_min+alfa)/2
        alfa_hi=(alfa+alfa_max)/2
    
        (fmin,~) = obj(x+alfa_min*d,pnum)               #                       
        (f,g) = obj(x+alfa*d,pnum)                      # Evaluate at the guess and at each bound.                       #
        (fmax,~) = obj(x+alfa_max*d,pnum)               #
        k=k+1
    end
    
    if dot(alfa*d,g-g0)<=0
        println("INDEFINITE")
    end
    
    return(alfa)
end

# Define problem to solve, call solver
pnum=9

p = Problem(pnum)
x0=p.x0
g0=p.grd(x0)
tic()
(xs, fs, grad,k)=bfgsmin(tomsobj,x0,pnum=pnum,maxIts=300,relTol=1e-6,absTol=1e-2)
toc()
gg0=grad/norm(g0);
println("f*=$fs")
println("g*=$grad")
println("g*./g0=$gg0")
println("its=$k")


