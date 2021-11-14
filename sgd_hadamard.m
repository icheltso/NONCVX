m=100;
n=200;
X = randn(m,n);
x0 = zeros(n,1);
x0(randperm(n,3)) = randn(3,1);
y = X*x0;

lam = norm(X'*y,'inf')/40;
f = @(beta) lam*norm(beta,1)+1/2*norm(X*beta-y)^2;


u = randn(n,1)*0.001;
v = randn(n,1)*0.001;
xinit = u.*v;
niter = 1000;
gamma = 0.001;
fvals_uv = [];
for j=1:niter
    gamma = 0.01/j.^0.51;
    k = randi(m);
    Xi = X(k,:);
    w = m* Xi'*(Xi*(u.*v) - y(k));
    uold = u;
    
    u = u - gamma*v.*w - lam*gamma*u;
    v = v - gamma*uold.*w - lam*gamma*v;
    fvals_uv(j) = f(u.*v);
    
end
%%
x = randn(n,1)*0.0001;
fvals_x = [];
for j=1:niter 
    gamma = 0.001/j.^0.51;
    k = randi(m);
    Xi = X(k,:);
    w = m* Xi'*(Xi*x - y(k));
    
    x = wthresh( x - gamma*w,'s',lam*gamma);
    fvals_x(j) = f(x);
    
end
clf
semilogy(fvals_uv);
hold on
semilogy(fvals_x);
%%
figure(1)
clf
hold on
stem(u.*v, 'bd')
stem(x, 'ro')
stem(x0, 'kx')

