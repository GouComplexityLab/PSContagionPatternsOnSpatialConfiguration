function dx = OdeSISystem(t,x,beta,nu,gamma)

dx = zeros(2,1);
dx(1) = - beta*(1+nu*x(2))*x(2)*x(1) + gamma * x(2);
dx(2) =   beta*(1+nu*x(2))*x(2)*x(1) - gamma * x(2);

end