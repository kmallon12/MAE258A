%% MAT258a Numerical Optimization
% Kevin Mallon
% Homework 2
% 10/7/15

clearvars
close all
clc

%% Load & assign data
load('admissions.mat')

m=length(admit);
q=sum(admit);

% x = [beta; alfa_gpa; alfa_gre; alfa_rank];
u = [ones(m,1), gpa, gre, rank];
y = admit;

%% Sort
[~,I] = sort([y,u],'descend');
ys=y(I(:,1));
us=u(I(:,1),:);

%% Solve problem (CVX)
% I had some trouble programming a first-order method (trying gradient
% descent), so I'm using CVX to 1) help debug code and 2) show the decision
% boundary. I tried this with and without rank as a component of u -- with
% rank gives much better results.

cvx_begin
    variables x(4)
    minimize(-sum(us(1:q,:)*x)+sum(log(1+exp(us*x))))
cvx_end

beta=x(1); alfa_gpa=x(2); alfa_gre=x(3); alfa_rank=x(4);

% Note: x*=[-3.4496,0.7770,0.0023,-0.5600]';

%% Plot Points, Boundary Line

figure(1);
plot(us(1:q,2),us(1:q,3),'b+');hold on;
plot(us(q+1:m,2),us(q+1:m,3),'ro');
plot(0:0.1:4, (1/alfa_gre)*(1/2-beta-alfa_gpa*(0:0.1:4)),'k','LineWidth',2)
axis([2 4 200 800])
title('Accept/Reject Points w/ Decision Boundary - CVX')
xlabel('GPA')
ylabel('GRE')
hold off

%% Cumulative Distribution
% Cumulative distribution for probability of admittance based on GPA and
% GRE scores. Used this check how reasonable the decision boundary is.

gpa_scale=linspace(2,4,25)';
gre_scale=linspace(200,800,25)';
cdf=zeros(length(gpa_scale),length(gre_scale));
for i=1:length(gpa_scale);
    for j=1:length(gre_scale);
        for k=1:m
            if gpa(k)<=gpa_scale(i) && gre(k)<=gre_scale(j)
                cdf(i,j)=cdf(i,j)+1/400;
            end
        end
    end
end

figure(2)
surf(gpa_scale,gre_scale,cdf)
xlabel('GPA');
ylabel('GRE');
zlabel('CDF');

%% Solve problem (Gradient Descent)
% To the best of my knowledge, this minimizes the negative log-likelihood 
% function. That said, it needs an absurdly small step size. I don't know
% why, but any L lower than about 5e7 causes the gradient to explode. Even
% starting near the optimal point requires millions of iterations at this
% rate. I haven't been able to run the routine to completion, but the
% gradient is continually skrinking and as far as I can tell it approaches
% the optimal point determined by the CVX solver.

L=5e7;  
errtol=1e-6;

% Initialize
x=zeros(4,500);     % Preallocate some steps.
x(:,1)=[-4;1;0;-1]; % Starting point.
grad_fx=[1;1;1;1];  % Used for starting the while loop only -- overwritten 
k=1;                % before the first step.

while norm(grad_fx)>=errtol
    xk=x(:,k);
    sumx1=-(sum(us(1:q,1))-sum(us(1:m,1)./(1+exp(-us(1:m,:)*xk))));
    sumx2=-(sum(us(1:q,2))-sum(us(1:m,2)./(1+exp(-us(1:m,:)*xk))));
    sumx3=-(sum(us(1:q,3))-sum(us(1:m,3)./(1+exp(-us(1:m,:)*xk))));
    sumx4=-(sum(us(1:q,4))-sum(us(1:m,4)./(1+exp(-us(1:m,:)*xk))));
    grad_fx=[sumx1;sumx2;sumx3;sumx4];
    x(:,k+1)=xk-(1/L)*grad_fx;
    k=k+1;
    
    if mod(k,1000)==0
        % Display some data so I know where the algorithm is at.
        disp(['k=',num2str(k)]);
        disp(['x=[',num2str(x(1,k)),',',num2str(x(2,k)),',',num2str(x(3,k)),',',num2str(x(4,k)),']']);
        disp(['norm(grad)=',num2str(norm(grad_fx))]);
    end
end

beta=x(1,k); alfa_gpa=x(2,k); alfa_gre=x(3,k); alfa_rank=(x(4,k));

%% Plot Points, Decision Boundary

figure(3);
plot(us(1:q,2),us(1:q,3),'b+');hold on;
plot(us(q+1:m,2),us(q+1:m,3),'ro');
plot(0:0.1:4, (1/alfa_gre)*(1/2-beta-alfa_gpa*(0:0.1:4)),'k','LineWidth',2)
axis([2 4 200 800])
title('Accept/Reject Points w/ Decision Boundary - CVX')
xlabel('GPA')
ylabel('GRE')
hold off