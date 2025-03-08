clear all
close all
clc

tic
%%
format long


beta = 0.05; nu = 2; gamma = 1/3; % illustration
ds_max = 30;

d_i = 1; 
%%
disp('=====================instability region on powelaw network=======================')
load powerlaw12eigenvalues_with_L=500 first_smallest_eigenvalues
netname = 'Netpowerlaw12L500';

%%

Lambda2_La = first_smallest_eigenvalues(2);

d_s_values = linspace(d_i,ds_max,10000+1);
critical_N_right_values1 = zeros(size(d_s_values));
critical_N_left_values1 = zeros(size(d_s_values));

N_values_pre = linspace(0,60,10000+1); 

% determine the min N for existence of positive equilibrium
if gamma/beta > 1/nu
    disp('backward case')
    Nmin = 2*sqrt(gamma/beta/nu) - 1/nu;
else
    disp('forward case')
    Nmin = gamma/beta;
end
N_values = N_values_pre(find(N_values_pre>=Nmin)); 

N_values = linspace(Nmin,70,10000+1); 

%% loop for N to determine spinodal (instability) region
pos_N = 1;
for N = N_values

    I = ( beta*(nu*N - 1) + sqrt((beta + beta*nu*N).^2 - 4*beta*nu*gamma) )/(2*beta*nu);
    I = real(I);
    S = N - I;
    f_S = beta*(1 + nu*I).*I;
    f_I = beta*(1 + 2*nu*I).*S - gamma;
    
    left_values = (d_i*f_S - d_s_values*f_I)./d_s_values/d_i;
    
    cha = Lambda2_La - left_values;
    pos_down = find(cha>0);
    
    if length(pos_down) >= 2
        critical_d_s_max1 = d_s_values(pos_down(end));
        critical_d_s_min1 = d_s_values(pos_down(1));

        critical_d_s_max1_values1(pos_N) = critical_d_s_max1;
        critical_d_s_min1_values1(pos_N) = critical_d_s_min1;
    else
        critical_d_s_max1_values1(pos_N) = nan;
        critical_d_s_min1_values1(pos_N) = nan;
    end
    pos_N = pos_N + 1;

end


%%
d_ss = linspace(1,ds_max,10000+1);

%%
abs_min_save = []; 
p_star_save = []; 
s0_save = []; i0_save = []; 
s_d_save = []; i_d_save = []; 
s_u_save = []; i_u_save = [];
tau_fu_save = [];  tau_zheng_save = []; tau_0_save = [];
tau_D_save = []; 


%% loop for ds to determine binodal region
for d_s = d_ss

    p_up = d_s*gamma/beta;
    p_low = 2*sqrt(d_s*d_i)*sqrt(gamma/beta/nu)-d_i/nu;
    
    min_cha = [];
    p_stars = linspace(p_low,p_up,10000+1);
    
    for p_star = p_stars   
        q_0 = p_star;
%       disp(['q0:    ' num2str(q_0,'%.10f') ])
        s0 = 1/2/d_s*(q_0 + p_star);
        i0 = 1/2/d_i*(p_star - q_0);
%       disp(['(s0, i0):    (' num2str(s0,'%.10f')  ', ' num2str(i0,'%.10f') ')' ])
        
        delta_q = (2*beta*d_i)^2 + 4*beta*nu*(2*d_i*beta*p_star + beta*nu*(p_star)^2 - 4*d_s*d_i*gamma);
        q_u = d_i/nu + (-1/2/beta/nu)*(delta_q)^(1/2);
%         disp(['q_u:    ' num2str(q_u,'%.10f') ])
        s_u = 1/2/d_s*(q_u + p_star);
        i_u = 1/2/d_i*(p_star - q_u);
%         disp(['(s_u, i_u):    (' num2str(s_u,'%.10f')  ', ' num2str(i_u,'%.10f') ')' ])
        
        q_d = d_i/nu - (-1/2/beta/nu)*(delta_q)^(1/2);
%         disp(['q_d:    ' num2str(q_d,'%.10f') ])
        s_d = 1/2/d_s*(q_d + p_star);
        i_d = 1/2/d_i*(p_star - q_d);
%         disp(['(s_d, i_d):    (' num2str(s_d,'%.10f')  ', ' num2str(i_d,'%.10f') ')' ])
        %% Hamiltonian function
        Hamilton = @(p,q) (d_s + d_i).*(q.^2.*(gamma./(4.*d_i) - (beta.*nu.*p_star.^2)/(16.*d_i.^2.*d_s)) - q.*((gamma.*p_star)./(2*d_i) - (beta.*p_star.^2.*((nu.*p_star)/(2.*d_i) + 1))./(4.*d_i.*d_s)) - (beta.*q.^3.*((nu.*p_star)/(2.*d_i) + 1))/(12.*d_i.*d_s) + (beta.*nu.*q.^4)./(32.*d_i.^2.*d_s));

        H0 = Hamilton(p_star,q_0);
        Hd = Hamilton(p_star,q_d);
        Hu = Hamilton(p_star,q_u);

        min_cha = [ min_cha ,H0-Hu] ;
    end
%% 

    [abs_min p] = min(abs(min_cha));
    abs_min_save = [ abs_min_save ,abs_min] ;
    % abs_min
    find_p_star=p_stars(p);
    disp(['d_s=: ' num2str(d_s,'%.10f') ',  abs_min=: ' num2str(abs_min,'%.10f')  ',  find_p_star=: ' num2str(find_p_star,'%.10f')  ]);

    p_star = find_p_star;
    q_0 = p_star;
    s0 = 1/2/d_s*(q_0 + p_star);
    i0 = 1/2/d_i*(p_star - q_0);

    delta_q = (2*beta*d_i)^2 + 4*beta*nu*(2*d_i*beta*p_star + beta*nu*(p_star)^2 - 4*d_s*d_i*gamma);

    q_u = d_i/nu + (-1/2/beta/nu)*(delta_q)^(1/2);
    s_u = 1/2/d_s*(q_u + p_star);
    i_u = 1/2/d_i*(p_star - q_u);

    q_d = d_i/nu - (-1/2/beta/nu)*(delta_q)^(1/2);
    s_d = 1/2/d_s*(q_d + p_star);
    i_d = 1/2/d_i*(p_star - q_d);

    p_star_save = [ p_star_save ,p_star] ;
    s0_save = [ s0_save ,s0] ;
    i0_save = [ i0_save ,s0] ;
    s_d_save = [ s_d_save ,s_d] ;
    i_d_save = [ i_d_save ,s_d] ;
    s_u_save = [ s_u_save ,s_u] ;
    i_u_save = [ i_u_save ,s_u] ;
    tau_0 = s0 + i0;
    tau_0_save = [ tau_0_save ,tau_0] ;
    tau_fu = s_d + i_d;
    tau_fu_save = [ tau_fu_save ,tau_fu] ;
    tau_zheng = s_u + i_u;
    tau_zheng_save = [ tau_zheng_save ,tau_zheng] ;

end

toc

data_name_small = [netname 'RegionDataSmall_beta=' dot2d(beta) '_nu=' dot2d(nu) '.mat' ' N_values critical_d_s_min1_values1 critical_d_s_max1_values1 tau_0_save tau_zheng_save d_ss'];
eval(['save ' data_name_small])
