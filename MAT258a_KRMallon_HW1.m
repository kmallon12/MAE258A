%% MAT258a_KRM_HW1.m
% Had some trouble with Julia, it was easier to just use Matlab for now.

clear all
close all
clc

%% Define
n=100;
A=randn(n,n);
b=randn(n,1);

%% 1-norm
cvx_begin
    variable x(n)
    minimize(0.5*sum_square(A*x-b)+1.5*norm(x,1));
cvx_end

%% plot
figure(1)
r=-2:.1/50:2;
hist(x,25);hold on;
xlim([-2,2]);ylim([0,40]);title('1-Norm Histogram');
plot(r,15*abs(r),'k','LineWidth',2);
hold off;

%% 2-norm
cvx_begin
    variable x(n);
    minimize(0.5*sum_square(A*x-b)+2*norm(x,2));
cvx_end

%% plot
figure(2)
r=-2:.1/50:2;
hist(x,25);hold on;
xlim([-2,2]);ylim([0,10]);title('2-Norm Histogram');
plot(r,2*r.^2,'k','LineWidth',2);
hold off;

%% Linear Deadzone
    
cvx_begin
    variable x(n);
    minimize(sum_square(A*x-b)+10*sum(max(abs(x)-0.5,0)));
cvx_end

%% plot
figure(3)
r=-2:.1/50:2;
hist(x,25);hold on;
xlim([-2,2]);ylim([0,10]);title('Linear Deadzone Histogram');
plot(r,10*max(abs(r)-0.5,0),'k','LineWidth',2);
hold off;

%% Log Barrier
    
cvx_begin
    variable x(n);
    minimize(sum_square(A*x-b)+1*(-sum_log(1-square_pos(x))));
cvx_end

%% plot
figure(4)
r=-0.9998:.01/50:0.9998;
hist(x,25);hold on;
xlim([-2,2]);ylim([0,10]);title('Log Barrier Histogram');
plot(r,.5*max(real(-log(r+1)),real(-log(r-1))),'k','LineWidth',2);
hold off;