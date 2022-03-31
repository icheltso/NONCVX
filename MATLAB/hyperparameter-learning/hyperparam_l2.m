% This code uses L-BFGS-B, downloaded from https://github.com/stephenbeckr/L-BFGS-B-C

clearvars

type = 'lsqr';
type = 'logreg';

dataset = 'randomGaussian';
dataset = 'libsvm';

name_list = {'australian','gisette','mushrooms','real-sim','covtype','news20_binary'};
name  = name_list{6};
switch dataset
    case 'randomGaussian'
        
        m = 80;
        n = 200;
        A = randn(m,n);
        b = randn(m,1);
        
        A_out = A+.7*randn(m,n);
        b_out = b+.7*randn(m,1);
        
        
    case 'libsvm'
        L = load(sprintf('datasets/%s_label.mat',name));
        Y = L.l;
        Y = Y(:);
        L = load(sprintf('datasets/%s_sample.mat',name));
        X = L.h;
        
        m = size(X,1);
        if m~= length(Y)
            X = X';
        end
        
        p = randperm(m);
        K = ceil(m/5);
        S = ceil(K*0.9);
        A = X(p(1:S),:);
        b = Y(p(1:S));
        
        A_out = X(p(1+S:K),:);
        b_out = Y(p(1+S:K));
        [m,n] = size(A);
        
        
end
%rescale to [-1,1]
if strcmp(type, 'logreg') %for logistic regression, labels should be -1,1
    b_out = b_out - min(b_out);
    b_out = 2*b_out/max(b_out)-1;
    
    b = b - min(b);
    b = 2*b/max(b)-1;
end
%%
switch type
    case 'logreg'
        % sum(log(1+exp(-x)))
        sumlogexp = @(x) sum( log(1+exp(-x(x>0))) ) +  sum( -x(x<=0) + log (exp(x(x<=0)) +1)  );
        %1/(1+exp(x))
        expit =  @(x) exp(-max(x,0))./( (exp(-max(x,0))+exp(min(x,0))) );
        % exp(x)/(1+exp(x)).^2
        expit2 = @(x) exp(min(x,0) -max(x,0))./(exp(-max(x,0))+exp(min(x,0))).^2;
        
        
        func_g = @(w,lambda) sumlogexp(b.*(A*w))+0.5*lambda*norm(w)^2;
        Gradx_g = @(w,lambda) A'* (-b.*expit(b.*(A*w))) +lambda*w;
        Gradf = @(w,lambda) deal( func_g(w,lambda), Gradx_g(w,lambda));
        
        
        x0 = randn(n,1);
        innerSolve  = @(lambda,x0) lbfgsb( @(w) Gradf(w,lambda), ...
                                        -inf(n,1), inf(n,1), ...
                                        struct('x0',x0,'printEvery', 100,  'maxIts', 100 ) );
        Gradx_f = @(x) A_out'* (-b_out.* expit(b_out.*(A_out*x)));
        
        Hessx_g = @(x,lambda,v) A'*( expit2(b.*(A*x)).*(A*v)) +lambda*v;% A'* spdiags((exp(b.*(A*x))./(1+exp(b.*(A*x))).^2),0,m,m)*A +lambda*speye(n);
        func_outer = @(w) sumlogexp(b_out.*(A_out*w));
    case 'lsqr'
        func_outer = @(x) norm(A_out *x - b_out)^2/2;
        Gradx_f =@(x) A_out'*(A_out*x - b_out);
        
        innerSolve = @(lambda,xinit) pcg(A'*A+lambda*speye(n), A'*b,1e-8,100,[],[],xinit);
        
        func_g = @(w,lambda) lambda*norm(w)^2/2 + norm(A*w-b)^2/2;
        Gradx_g = @(w,lambda) A'* (A*w-b) +lambda*w;
        
        Hessx_g = @(x,lambda,v) A'*(A*v)+lambda*v;
        
end


%% Use implicit formula for outer gradient
tic
niter = 50; %max iterations
yinit = 1; %initial value
tau = .1; %stepsize

%%
y = yinit;
fval = [];
gval = [];
x = randn(n,1);
for i =1:niter
    
    x = innerSolve(exp(y),x);
    dx_f = Gradx_f(x);
    dyx_g = exp(y)*x;
    dxx_g = @(v) Hessx_g(x,exp(y),v);
    
    grad = - dx_f'*pcg(dxx_g,dyx_g); %PCG to invert Hessian
    y = y - tau*grad;
    if norm(grad)<1e-8
        break
    end
    
    fval(i) = func_outer(x);
    gval(i) = norm(grad);
end

plot(fval)
y
y0 = y;
time1 = toc
%% Use quasi-newton for outer gradient
tic

y=yinit;
fval2 = [];
gval2 = [];
x = innerSolve(exp(y),randn(n,1));

%take 1 gradient step to start
dx_f = Gradx_f(x);
dyx_g = exp(y)*x;
dxx_g = @(v) Hessx_g(x,exp(y),v);
grad = - dx_f'*pcg(dxx_g,dyx_g);
% grad = - dx_f'*(Solve(exp(y)) - Solve(exp(y-0.1)))/0.1;
y = y - tau*grad;

gamma = 0.1;
for i =1:niter
    
    xm = x;
    x = innerSolve(exp(y),x);
    
    dx_f = Gradx_f(x);
    dyx_g = exp(y)*x;
    
    %rank-1 + diagonal approximation of Hessian inverse
    s = xm-x;
    z = Gradx_g(xm,exp(y));
    t = sum(s.*z)/norm(z)^2;
    u = s-gamma*t*z;
    bk = gamma*t*dyx_g;
    if i>2 && sum(u.*z)>10^-8*norm(z)*norm(u)
        bk = bk+ u*(u'*dyx_g)/sum(u.*z);
    end
    
    grad = - dx_f'*bk;
    y = y - tau*grad;
    
    
    fval2(i) = func_outer(x);
    gval2(i) = norm(grad);
    if norm(grad)<1e-8
        break
    end
end

% plot(fval2)
y_sr1 = y
time2 = toc

%% cross validation curve

yvals = linspace(-6,2,50);

cv= [];
x = randn(n,1);
for i =1:length(yvals)
    lambda = exp(yvals(i));
    x = innerSolve(lambda,x);
    cv(i) = func_outer(x);
end
%%
clf
plot(yvals,cv, 'x')
hold on
semilogx([(y),(y)],[min(cv),max(cv)], 'r', 'linewidth',2)
semilogx([(y0),(y0)],[min(cv),max(cv)], 'g', 'linewidth',1)
% xlim([1e-5,1e2])

