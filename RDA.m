m=100;
n=200;
X = randn(m,n);
x0 = zeros(n,1);
x0(randperm(n,3)) = randn(3,1);
y = X*x0;

lam = norm(X'*y,'inf')/4;
f = @(beta) lam*norm(beta,1)+1/2*norm(X*beta-y)^2;


%%
niter = 50000;
x = 0*randn(n,1)*0.0001;
fvals_x = [];
g= zeros(n,1);
rho = .005;
gamma = 5000;
for t=1:niter 
    k = randi(m);
    Xi = X(k,:);
    w = m* Xi'*(Xi*x - y(k));
    
    g = (t-1)/t*g+1/t*w;
    
    la_rda = lam + rho/sqrt(t);
    x = -max(abs(g) - la_rda,0).*sign(g)*sqrt(t)/gamma;
    
    fvals_x(t) = f(x);
    
end
clf
semilogy(fvals_x);
%%
figure(1)
clf
hold on
stem(x, 'ro')
stem(x0, 'kx')
legend('RDA', 'true')
