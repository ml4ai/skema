clear all

% define integration interval
t0 = 0;
tfinal = 2.*365;

% Ontario population, 14.57M 2019
N = 14570000;

% define parameters
% these are infection rates without lockdown
beta = [3.3e-9;5.5e-9;7.6e-9]*N; % infection rate, for susceptibles
% beta_v1 = [0.3;0.5].*beta; % infection rate, first dose
% beta_v2 = 0.05*beta; % infection rate, both doses
% https://www.gov.uk/government/news/vaccines-highly-effective-against-b-1-617-2-variant-after-2-doses
beta_v1 = [0.2 0.5 0.67; 0.2 0.5 0.67].*[beta';beta']; % infection rate, first dose; row, for a given vaccine type; column, for a given variant
beta_v2 = [0.05 0.07 0.12; 0.05 0.34 0.4].*[beta';beta']; % infection rate, both doses
beta_R  = 0.05*beta; % infection rate after recovery
ai_beta_ratio = [3; 3; 3];  % asymptomatic vs. symptomatic infectivity ratios
% vaccination rates now given as function of time in get_vaccine_rates.m
% vr1  = 1e-3; % vaccination rates (per day)
% vr2  = 1/21; % 21 days delay
gamma = 1/28; % recovery rate
nu_v1 = 2*0.25/182; % loss of immunity, first dose (6 months)
nu_v2 = 2*0.125/365; % loss of immunity, both doses (1 year)
nu_R  = 2*0.125/365; % loss of immunity, recovered (1 year)
ai    = [0.5; 0.5; 0.5];  % fraction of asymtomatic primary infections
ai_V  = [0.85; 0.85; 0.85];  % fraction of asymtomatic infections after vacciation
ai_R  = [0.85; 0.85; 0.85];  % fraction of asymtomatic infections after recovery from another variant
mu    = 109019/N/365;  % natural death rate (109019 in 14.5 M in 2018-2019)
mu_I  = 1.75*[9255/555927*gamma; 1.6*9255/555927*gamma; 1.8*9255/555927*gamma];  % COVID mortaolity rate, (9255 deaths for 555927 total cases)
mu_IV = 0.15*mu_I;  % vaccine reduces mortality rate

% parameters for new killer variant, will replace wild-type after fall 2021
new_beta    = beta(3); %2.2*beta(1);
new_beta_v1 = [0.5; 0.5]*new_beta;
new_beta_v2 = [0.2; 0.2]*new_beta;
new_beta_R  = 0.05*new_beta;
new_ai      = 0.8;
t_new_voc   = daysact('1-jan-2020','1-sep-2022');

% define initial population fractions
I0   = [1e-6;0;0];  % infected
A0   = [0;0;0];
S0   = 1-sum(I0+A0);  % original susceptible
SVR0 = 0;  % lost immunity after vaccination or recovery
V10  = [0;0];  % one-dose vaccination
V20  = [0;0];  % fully vaccinated
IV0  = [0;0;0];  % infected even with vaccination
IR0  = [0;0;0];  % infected again after recovery from a different variant
AR0  = [0;0;0];  % asymptomatic infection after recovery from a different variant
R0   = [0;0;0];  % recovered
R20  = 0;  % recovered after getting both variants
y0 = [S0;SVR0;V10;V20;I0;IV0;IR0;A0;AR0;R0;R20];

[t,y] = ode15s(@(t,y) svair(t,y,beta,beta_v1,beta_v2,beta_R,ai_beta_ratio,gamma,nu_v1,nu_v2,nu_R,ai,ai_V,ai_R,mu,mu_I,mu_IV,new_beta,new_beta_v1, ...
    new_beta_v2,new_beta_R,new_ai,t_new_voc),[t0 tfinal],y0);
S   = y(:,1);
SVR = y(:,2);
V1PF= y(:,3); V1AZ=y(:,4); 
V2PF= y(:,5); V2AZ=y(:,6);
IP  = y(:,7);  IA  = y(:,8);  ID  = y(:,9);
IPV = y(:,10); IAV = y(:,11); IDV = y(:,12);
IPR = y(:,13); IAR = y(:,14); IDR = y(:,15);
AP  = y(:,16); AA  = y(:,17); AD  = y(:,18);
APR = y(:,19); AAR = y(:,20); ADR = y(:,21);
RP  = y(:,22); RA  = y(:,23); RD  = y(:,24);
R2  = y(:,25);

% figure(1), clf
% subplot(3,3,1)
% plot(t,S); hold on
% plot(t,SVR);
% legend('S','SVR')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(3,3,2)
% plot(t,V1PF); hold on
% plot(t,V1AZ);
% plot(t,V2PF);
% plot(t,V2AZ);
% legend('V1PF','V1AZ','V2PF','V2AZ')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(3,3,3)
% plot(t,IP); hold on
% plot(t,IA);
% plot(t,ID);
% legend('I^P','I^A','I^D')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(3,3,4)
% plot(t,IPV); hold on
% plot(t,IDV);
% plot(t,IAV);
% legend('I^P_V','I^A_V','I^D_V')
% 
% subplot(3,3,5)
% plot(t,IPR); hold on
% plot(t,IDR);
% plot(t,IAR);
% legend('I^P_R','I^A_R','I^D_R')
% 
% subplot(3,3,6)
% plot(t,AP); hold on
% plot(t,AD);
% plot(t,AA);
% legend('A^P','A^A','A^D')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(3,3,7)
% plot(t,APR); hold on
% plot(t,ADR);
% plot(t,AAR);
% legend('A^P_R','A^A_R','A^D_R')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(3,3,8)
% plot(t,RP); hold on
% plot(t,RD);
% plot(t,RA);
% plot(t,R2);
% legend('R^P','R^A','R^D','R2');
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})

% compute number of new infections
tn  = length(t);
dIPdt = zeros(1,tn);
dIAdt = zeros(1,tn);
dIDdt = zeros(1,tn);
dIPVdt = zeros(1,tn);
dIAVdt = zeros(1,tn);
dIDVdt = zeros(1,tn);
dAPdt = zeros(1,tn);
dAAdt = zeros(1,tn);
dADdt = zeros(1,tn);
Deadtot = zeros(1,tn);
DeadPtot = zeros(1,tn);
DeadDtot = zeros(1,tn);
DeadAtot = zeros(1,tn);
V1tot = zeros(1,tn);
V2tot = zeros(1,tn);
beta0    = beta;
beta_v10 = beta_v1;
beta_v20 = beta_v2;
beta_R0  = beta_R;
for nt = [2:tn]
%     yp   = svir(t(nt),y(nt,:)',beta,beta_v1,beta_v2,beta_R,gamma,nu_v1,nu_v2,nu_R,mu,mu_I,mu_IV);
    beta_scale = get_beta (t(nt));
    beta       = beta0*beta_scale;
    beta_v1    = beta_v10*beta_scale;
    beta_v2    = beta_v20*beta_scale;
    beta_R     = beta_R0*beta_scale;   
    dIPdt(nt)   = (1-ai(1))*(+ beta(1).*S(nt).*(IP(nt)+IPV(nt)+IPR(nt)+ai_beta_ratio(1)*(AP(nt)+APR(nt))) + beta(1).*SVR(nt).*(IP(nt)+IPV(nt)+IPR(nt)+ai_beta_ratio(1)*(AP(nt)+APR(nt))));
    dIAdt(nt)   = (1-ai(2))*(+ beta(2).*S(nt).*(IA(nt)+IAV(nt)+IAR(nt)+ai_beta_ratio(2)*(AA(nt)+AAR(nt))) + beta(2).*SVR(nt).*(IA(nt)+IAV(nt)+IAR(nt)+ai_beta_ratio(2)*(AA(nt)+AAR(nt))));
    dIDdt(nt)   = (1-ai(3))*(+ beta(3).*S(nt).*(ID(nt)+IDV(nt)+IDR(nt)+ai_beta_ratio(3)*(AD(nt)+ADR(nt))) + beta(3).*SVR(nt).*(ID(nt)+IDV(nt)+IDR(nt)+ai_beta_ratio(3)*(AD(nt)+ADR(nt))));
    dIPVdt(nt)  = (1-ai_V(1))*(+ sum(beta_v1(:,1).*[V1PF(nt),V1AZ(nt)]')*(IP(nt)+IPV(nt)+IPR(nt)+ai_beta_ratio(1)*(AP(nt)+APR(nt))) + sum(beta_v2(:,1).*[V2PF(nt),V2AZ(nt)]').*(IP(nt)+IPV(nt)+IPR(nt)+ai_beta_ratio(1)*(AP(nt)+APR(nt)))');
    dIAVdt(nt)  = (1-ai_V(2))*(+ sum(beta_v1(:,2).*[V1PF(nt),V1AZ(nt)]')*(IA(nt)+IAV(nt)+IAR(nt)+ai_beta_ratio(2)*(AA(nt)+AAR(nt))) + sum(beta_v2(:,2).*[V2PF(nt),V2AZ(nt)]').*(IA(nt)+IAV(nt)+IAR(nt)+ai_beta_ratio(2)*(AA(nt)+AAR(nt)))');
    dIDVdt(nt)  = (1-ai_V(3))*(+ sum(beta_v1(:,3).*[V1PF(nt),V1AZ(nt)]')*(ID(nt)+IDV(nt)+IDR(nt)+ai_beta_ratio(3)*(AD(nt)+ADR(nt))) + sum(beta_v2(:,3).*[V2PF(nt),V2AZ(nt)]').*(ID(nt)+IDV(nt)+IDR(nt)+ai_beta_ratio(3)*(AD(nt)+ADR(nt)))');
    dAPdt(nt)   = ai(1)*(+ beta(1).*S(nt).*(IP(nt)+IPV(nt)+IPR(nt)+ai_beta_ratio(1)*(AP(nt)+APR(nt))) + beta(1).*SVR(nt).*(IP(nt)+IPV(nt)+IPR(nt)+ai_beta_ratio(1)*(AP(nt)+APR(nt))));
    dAAdt(nt)   = ai(2)*(+ beta(2).*S(nt).*(IA(nt)+IAV(nt)+IAR(nt)+ai_beta_ratio(1)*(AA(nt)+AAR(nt))) + beta(2).*SVR(nt).*(IA(nt)+IAV(nt)+IAR(nt)+ai_beta_ratio(1)*(AA(nt)+AAR(nt))));
    dADdt(nt)   = ai(3)*(+ beta(3).*S(nt).*(ID(nt)+IDV(nt)+IDR(nt)+ai_beta_ratio(3)*(AD(nt)+ADR(nt))) + beta(3).*SVR(nt).*(ID(nt)+IDV(nt)+IDR(nt)+ai_beta_ratio(3)*(AD(nt)+ADR(nt))));
    
    Deadtot(nt) = Deadtot(nt-1) + (t(nt)-t(nt-1))*(mu_I(1)*IP(nt)+mu_I(2)*IA(nt)+mu_I(3)*ID(nt)+mu_IV(1)*IPV(nt)+mu_IV(2)*IAV(nt)+mu_IV(3)*IDV(nt));
    DeadPtot(nt) = DeadPtot(nt-1) + (t(nt)-t(nt-1))*(mu_I(1)*IP(nt)+mu_IV(1)*IPV(nt));
    DeadAtot(nt) = DeadAtot(nt-1) + (t(nt)-t(nt-1))*(mu_I(2)*IA(nt)+mu_IV(2)*IAV(nt));
    DeadDtot(nt) = DeadDtot(nt-1) + (t(nt)-t(nt-1))*(mu_I(3)*ID(nt)+mu_IV(3)*IDV(nt));
        
    % compute total vaccinated, including those who subsequently lose their
    % immunity
    [vr1, vr2] = get_vaccine_rate (t(nt));
    V1tot(nt)  = V1tot(nt-1) + (t(nt)-t(nt-1))*sum(vr1).*(S(nt)+AP(nt)+AA(nt)+AD(nt));  % at least one dose
    V2tot(nt)  = V2tot(nt-1) + (t(nt)-t(nt-1))*(vr2(1)*V1PF(nt-1)+vr2(2)*V1AZ(nt-1));  % fully vaccinated
end
tnew  = [t0:tfinal];
dPnew = pchip(t,N*(dIPdt+dIPVdt+dAPdt),tnew);
dAnew = pchip(t,N*(dIAdt+dIAVdt+dAAdt),tnew);
dDnew = pchip(t,N*(dIDdt+dIDVdt+dADdt),tnew);

RP2    = dPnew./pchip(t,N*(IP+IPV+AP)',tnew)/gamma;
RA2    = dAnew./pchip(t,N*(IA+IAV+AA)',tnew)/gamma;
RD2    = dDnew./pchip(t,N*(ID+IDV+AD)',tnew)/gamma;

% figure(2), clf
% subplot(2,3,1)
% plot(tnew,movmean(dPnew,7)); hold on
% plot(tnew,movmean(dAnew,7));
% plot(tnew,movmean(dDnew,7));
% plot(tnew,movmean(dPnew+dAnew+dDnew,7));
% % ii = find(t<365*2);
% % plot(t(ii),N*(dIPdt(ii)+dIPVdt(ii))), hold on
% % plot(t(ii),N*(dIAdt(ii)+dIAVdt(ii)))
% % plot(t(ii),N*(dIDdt(ii)+dIDVdt(ii)))
% % plot(t(ii),N*(dIPdt(ii)+dIPVdt(ii)+dIAdt(ii)+dIAVdt(ii)+dIDdt(ii)+dIDVdt(ii)))
% %plot(t,N*(dIDdt+dIDVdt))
% legend('New Primary','New Alpha','New Delta','New total')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(2,3,2)
% plot(t,cumsum(N*(dDeaddt))); 
% legend('Total deaths')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(2,3,3)
% plot(t,cumsum(N*(dIPdt+dIPVdt))); hold on
% plot(t,cumsum(N*(dIAdt+dIAVdt)));
% plot(t,cumsum(N*(dIDdt+dIDVdt)));
% plot(t,cumsum(N*(dIPdt+dIPVdt+dIAdt+dIAVdt+dIDdt+dIDVdt)));
% legend('Total Primary','Total Alpha','Total Delta','Total')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(2,3,4)
% plot(t,cumsum(N*(dIPdt+dIPVdt+dAPdt))); hold on
% plot(t,cumsum(N*(dIAdt+dIAVdt+dAAdt)));
% plot(t,cumsum(N*(dIDdt+dIDVdt+dADdt)));
% plot(t,cumsum(N*(dIPdt+dIPVdt+dAPdt+dIAdt+dIAVdt+dAAdt+dIDdt+dIDVdt+dADdt)));
% legend('Total Primary','Total Alpha','Total Delta','Total')
% xticks([90 181 273 365 455 546 638]);
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(2,3,5)
% plot(tnew,movmean(RP2,30)); hold on
% plot(tnew,movmean(RA2,30));
% plot(tnew,movmean(RD2,30));
% xticks([90 181 273 365 455 546 638]);
% legend('Primary R0','Alpha R0','Delta R0')
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep'})
% 
% subplot(2,3,6)
% plot(t,V1tot); hold on
% plot(t,V2tot);
% xticks([90 181 273 365 455 546 638 739]);
% legend('>= 1 dose','2 doses')
% xticklabels({'31 Mar','30 Jun','30 Sep','31 Dec','31 Mar','30 Jun','30 Sep','31 Dec'})

save output
plot_data