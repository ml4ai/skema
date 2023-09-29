function yp = svair(t,y,beta,beta_v1,beta_v2,beta_R,ai_beta_ratio,gamma,nu_v1,nu_v2,nu_R,ai,ai_V,ai_R,mu,mu_I,mu_IV, ...
    new_beta,new_beta_v1,new_beta_v2,new_beta_R,new_ai,t_new_voc)

% retrieve current populations
nv   = 2;  % number of viruses simulated
ind  = 1;
S    = y(ind); ind=ind+1;  % original susceptible
SVR  = y(ind); ind=ind+1;  % lost immunity after vaccination or recovery
V1   = y(ind:ind+1); ind=ind+2;  % one-dose vaccination
V2   = y(ind:ind+1); ind=ind+2;  % fully vaccinated
I    = y(ind:ind+nv); ind=ind+nv+1;  % infected
IV   = y(ind:ind+nv); ind=ind+nv+1;  % infected even with vaccination
IR   = y(ind:ind+nv); ind=ind+nv+1;  % infected again after recovery from a different variant
A    = y(ind:ind+nv); ind=ind+nv+1;  % asymptomatic infections
AR   = y(ind:ind+nv); ind=ind+nv+1;  % asymptomatic infections after recovery from a different variant
R    = y(ind:ind+nv); ind=ind+nv+1;  % recovered
R2   = y(ind); ind=ind+1;  % recovered after getting both variants

% get time-dependent parameters
[vr1, vr2] = get_vaccine_rate (t);
if(t>=t_new_voc)  % switch WT to new VOC
    beta(1)      = new_beta;
    beta_v1(:,1) = new_beta_v1;
    beta_v2(:,1) = new_beta_v2;
    beta_R(1)    = new_beta_R;
    ai(1)        = new_ai;
end
beta_scale = get_beta (t);
beta       = beta*beta_scale;
beta_v1    = beta_v1*beta_scale;
beta_v2    = beta_v2*beta_scale;
beta_R     = beta_R*beta_scale;

% total infectious population
I_total = I+IV+IR+ai_beta_ratio.*(A+AR);
% need the following to compute infection of recovered from another variant
mm = ones(nv+1)-diag(ones(1,nv+1));
mv = mm.*repmat(R,1,nv+1);
Rv = sum(mv)';

% compute time derivatives
dSdt   = - sum(beta.*S.*(I_total)) - sum(vr1.*S) + mu*(1-S);
dSVRdt = + nu_v1.*sum(V1) + nu_v2.*sum(V2) + sum(nu_R.*R) + nu_R.*R2 - sum(beta.*SVR.*(I_total)) - mu*SVR;
dV1dt  = + vr1.*(S+sum(A)) - vr2.*V1 - nu_v1.*V1 - sum(beta_v1.*(V1*(I_total)'),2) - mu*V1;
dV2dt  = + vr2.*V1 - nu_v2.*V2 - sum(beta_v2.*(V2.*(I_total)'),2) - mu*V2;
dIdt   = (1-ai).*(+ beta.*S.*(I_total) + beta.*SVR.*(I_total)) - gamma.*I - mu_I.*I;
dIVdt  = (1-ai_V).*(+ sum(beta_v1.*(V1*(I_total)'))' + sum(beta_v2.*(V2.*(I_total)'))') - gamma.*IV - mu_IV.*IV;
% dIRdt  = (1-ai_R).*(+ beta_R.*flip(R).*(I_total)) - gamma.*IR - mu*IR;
dIRdt  = (1-ai_R).*(+ beta_R.*Rv.*(I_total)) - gamma.*IR - mu*IR;
dAdt   = ai.*(+ beta.*S.*(I_total) + beta.*SVR.*(I_total)) + ai_V.*(sum(beta_v1.*(V1*(I_total)'))' + sum(beta_v2.*(V2.*(I_total)'))') - sum(vr1).*A - gamma.*A - mu.*A;
% dARdt  = ai_R.*(+ beta_R.*flip(R).*(I_total)) - gamma.*AR - mu*AR;
dARdt  = ai_R.*(+ beta_R.*Rv.*(I_total)) - gamma.*AR - mu*AR;
% dRdt   = + gamma.*(I_total) - nu_R.*R - beta_R.*flip(R).*(I_total) - mu*R;
dRdt   = + gamma.*(I_total) - nu_R.*R - beta_R.*Rv.*(I_total) - mu*R;
dR2dt  = + sum(gamma.*IR) - nu_R.*R2 - mu*R2;

% simulate mutation of WT into variants and importation
if (t>315 & t<365)  % alpha appears
    dWTtoA  = 0e-5*dIdt(1);  % don't assume mutation
    dIdt(1) = dIdt(1)-dWTtoA;
    dAdt(2) = dAdt(2)+dWTtoA + 50/14570000; %((t-300)/60)*80/14570000;  % people enter Ontario with alpha

elseif (t>385 & t<445)
%     dAdt(3) = dAdt(3)+((t-385)/60)*40/14570000;  % delta was born
    dAdt(3) = dAdt(3)+25/14570000;  % delta was born
end

yp = [dSdt;dSVRdt;dV1dt;dV2dt;dIdt;dIVdt;dIRdt;dAdt;dARdt;dRdt;dR2dt];


