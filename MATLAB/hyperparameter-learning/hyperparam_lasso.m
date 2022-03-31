% This code uses L-BFGS-B, downloaded from https://github.com/stephenbeckr/L-BFGS-B-C


clearvars
type = 'ridge';
% type = 'logreg';

dataset = 'libsvm';
dataset = 'randomGaussian';

name_list = {'dexter','cadata','a8a','australian','gisette','mushrooms','real-sim','covtype'};
name  = name_list{7};
switch dataset
    case 'randomGaussian'

        m = 50;
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
        Afun = L.h;

        m = size(Afun,1);
        if m~= length(Y)
            Afun = Afun';
        end
        S = ceil(m*0.9);
        p = randperm(ceil(m));
        A = Afun(p(1:S),:);
        b = Y(p(1:S));

        A_out = Afun(p(1+S:end),:);
        b_out = Y(p(1+S:end));
        [m,n] = size(A);      
        
       
end
 %rescale to [-1,1]
 if strcmp(type, 'logreg') %for logistic regression, labels should be -1,1
     b_out = b_out - min(b_out);
     b_out = 2*b_out/max(b_out)-1;
     
     b = b - min(b);
     b = 2*b/max(b)-1;
 end

%% define helper functions
prod = @(uv) uv(n+1:end).*uv(1:n);
X1 = @(uv) uv(1:n);
X2 = @(uv) uv(n+1:end);
switch type
    case 'logreg'     
        
        %useful function for logistic regression (to prevent overflow)
        % sum(log(1+exp(-x)))
        sumlogexp = @(x) sum( log(1+exp(-x(x>0))) ) +  sum( -x(x<=0) + log (exp(x(x<=0)) +1)  );
        %1/(1+exp(x))
        expit =  @(x) exp(-max(x,0))./( (exp(-max(x,0))+exp(min(x,0))) );
        % exp(x)/(1+exp(x)).^2
        expit2 = @(x) exp(min(x,0) -max(x,0))./(exp(-max(x,0))+exp(min(x,0))).^2;


        func_outer = @(uv) sumlogexp(b_out.*(A_out*prod(uv)));       
        Gradx_f = @(uv) [X2(uv); X1(uv)].*repmat( A_out'* (-b_out.* expit(b_out.*(A_out* prod(uv) ))),2,1);
        
        
        func_g = @(uv,lambda) sumlogexp(b.*(A*prod(uv)))+0.5*lambda*norm(uv)^2;
        Gradx_g = @(uv,lambda) [X2(uv); X1(uv)].*repmat( A'* ( -b.*expit( b.*(A*prod(uv)) ) )  ,2,1)+lambda*uv;
        
        zvec = @(u,v)  expit2(b.*(A*(u.*v)));
        
        Afun = @(x) A*x;
        Atfun = @(x) A'*x;

        duu = @(u,v,w,lambda) v.* Atfun(zvec(u,v).*Afun(v.*w))+ lambda*w;
        dvv = @(u,v,w,lambda) u.* Atfun(zvec(u,v).*Afun(u.*w))+ lambda*w;
        dudv = @(u,v,w)  Atfun(-b.*expit(b.*(Afun(u.*v)))).*w + v.*Atfun(zvec(u,v).*Afun(u.*w));
        
        
    case 'ridge'
        func_outer = @(uv) norm(A_out *prod(uv) - b_out)^2/2;
        Gradx_f =@(uv) [X2(uv); X1(uv)].*repmat( A_out'*(A_out*prod(uv) - b_out), 2,1);
        
        
        func_g = @(uv,lambda) lambda*norm(uv)^2/2+ norm(A*prod(uv) - b)^2/2;               
        Gradx_g = @(uv,lambda) [X2(uv); X1(uv)].*repmat(A'* (A*prod(uv)-b),2,1)  +lambda*uv;       
        
        Afun = @(x) A*x;
        Atfun = @(x) A'*x;

        duu = @(u,v,w,lambda) v.* Atfun(Afun(v.*w))+ lambda*w;
        dvv = @(u,v,w,lambda) u.* Atfun(Afun(u.*w))+ lambda*w;
        dudv = @(u,v,w)  Atfun(Afun(u.*v)-b).*w + v.*Atfun(Afun(u.*w));
        
        
end
Hessx_g = @(uv,lambda,w) [duu(X1(uv),X2(uv),X1(w),lambda) + dudv(X1(uv),X2(uv),X2(w)); dudv(X2(uv),X1(uv),X1(w))+ duu(X1(uv),X2(uv),X2(w),lambda) ];
        

innerSolve = @(lambda, uv_init)  GD_innersolve( @(x) Gradx_g(x, lambda), uv_init,.01,100);
innerSolve = @(lambda,uv_init) LBFGS_solve(uv_init,@(uv) func_g(uv,lambda), @(uv) Gradx_g(uv,lambda),n);


%%
niter = 500;
yinit = 1;
tau = .01;


%%
y = yinit;
fval = [];
gval = [];
uv = randn(2*n,1)*1e-3;

for i =1:niter
    lambda = exp(y);
    uv = innerSolve(lambda,uv);
    dx_f = Gradx_f(uv);
    dyx_g = exp(y)*uv;

    grad = -dx_f'*pcg(@(w) Hessx_g(uv,lambda,w) , dyx_g);
    
    y = y - tau*grad;      
    
    fval(i) = func_outer(uv);
    gval(i) = norm(grad);
    if norm(grad)<1e-8
        break
    end
end

plot(fval)
y

y0 = y;
%% quasi-newton

gamma = 0.1;
y = yinit;

lambda = exp(y);
uv0 = randn(n*2,1);
uv = innerSolve(lambda,uv0);
dx_f = Gradx_f(uv);
dyx_g = exp(y)*uv;
grad = -dx_f'*pcg(@(w) Hessx_g(uv,lambda,w) , dyx_g);

y = y - tau*grad;


fval2 = [];
gval2 = [];
for i =1:niter
    
    uvm =  uv;
    
    uv = innerSolve(lambda,uv);
    
    dx_f = Gradx_f(uv);
    dyx_g = exp(y)*uv;

    
    s = uvm - uv;        
    z = Gradx_g(uvm,exp(y));      
    
    t = sum(s.*z)/norm(z)^2;
    bk = gamma*t*dyx_g;
    if i>2 && sum((s-gamma*t*z).*z  )>10^-8*norm(z)*norm(s-gamma*t*z)
        bk = bk+ (s-gamma*t*z)*((s-gamma*t*z)'*dyx_g)/sum((s-gamma*t*z).*z  );
    end
    
    grad = - dx_f'*bk;
    y = y - tau*grad;
          
    fval2(i) = func_outer(uv);
    gval2(i) = norm(grad);
    if norm(grad)<1e-8
        break
    end
end
y_sr1 = y
plot(fval2)





%%
yvals = linspace(-7,8,50);
cv= [];
for i =1:length(yvals)
    y = yvals(i);
    lambda = exp(y);
    uv = innerSolve(lambda,randn(n*2,1));
    cv(i) = func_outer(uv);
end
%%
clf
plot(yvals,cv, 'x')
hold on
plot([y_sr1,y_sr1],[min(cv),max(cv)], 'r', 'linewidth',2)
plot([y0,y0],[min(cv),max(cv)], 'g', 'linewidth',1)
legend('cv', '0mem-SR1', 'implicit')

xlabel('y')
ylabel('function value')
%%


function uv = LBFGS_solve(uv0,mfunc,mgrad,n)
        
  
Gradf = @(uv)  deal(mfunc(uv), mgrad(uv));


warning off;
lb = -inf(n*2,1);
ub = inf(n*2,1);
opts    = struct('x0',uv0,'printEvery', -1, 'm', 5, 'maxIts', 100 );
uv =  lbfgsb(  Gradf, lb, ub, opts );
u = uv(1:n);
v = uv(n+1:end);

% %force the sign of u and v to be consistent
u = sign(u.*v).*abs(u);
v = abs(v);
uv = [u;v];
end


function x = GD_innersolve(Gradx_g,xinit,tau, niter)
        
x = xinit;
for i =1:niter
dx = Gradx_g(x);
x=x-tau*dx;

end
% %force the sign of u and v to be consistent
% u = x(1:end/2);
% v = x(1+end/2:end);
% u = sign(u.*v).*abs(u);
% v = abs(v);
% x = [u;v];
end