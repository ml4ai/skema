dateticks = [60 151 243 334 425 516 608 699 790];
datelabels = {'Mar','Jun','Sep','Dec','Mar','Jun','Sep','Dec','Mar','Jun'};

load output

% population fractions
Stot = S+SVR;
Vtot = V1PF+V1AZ+V2PF+V2AZ;
Itot = IP+IPV+IPR+IA+IAV+IAR+ID+IDV+IDR;
Atot = AP+APR+AA+AAR+AD+ADR;
Rtot = RP+RA+RD+R2;

figure(2),clf
subplot(3,3,9)  % infections after vaccination
plot(t,ones(length(t),1),'-r','LineWidth',2), hold on
plot(t,Vtot+Itot+Atot+Rtot,'-b','LineWidth',2)
plot(t,Itot+Atot+Rtot,'-g','LineWidth',2)
plot(t,Itot+Atot,'-k','LineWidth',2)
shade(t,ones(length(t),1),t,Vtot+Itot+Atot+Rtot,t,Itot+Atot+Rtot,t,Itot+Atot,t,0*Rtot,'FillType',[1 2;2 1;2 3;3 4;4 5],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0; 0 0 0],'FillAlpha',[1 1 1 1 1]);
legend({'S','V','R','I+A'},'Fontsize',14);
axis tight
set(gca,'Fontsize',14)
xticks(dateticks);
xticklabels(datelabels)
ylabel('Population fraction','Fontsize',16)
title('(i)','Fontsize',16)

S = N*S; SVR = N*SVR;
V1PF = N*V1PF; V1AZ = N*V1AZ;
V2PF = N*V2PF; V2AZ = N*V2AZ;
IP = N*IP; IA = N*IA; ID = N*ID;
IPV = N*IPV; IAV = N*IAV; IDV = N*IDV;
IPR = N*IPR; IAR = N*IAR; IDR = N*IDR;
AP = N*AP; AA = N*AA; AD = N*AD;
APR = N*APR; AAR = N*AAR; ADR = N*ADR;
RP = N*RP; RA = N*RA; RD = N*RD;

% figure(2), clf

subplot(3,3,1)  % susceptibles
plot(t,S+SVR,'-r','LineWidth',2), hold on
plot(t,SVR,'-b','LineWidth',2)
shade(t,S+SVR,t,SVR,t,0*SVR,'FillType',[1 2;2 1;2 3],'FillColor',[1 0 0;0 1 0;0 0 1],'FillAlpha',[1 1 1]);
legend({'S','S_{VR}'},'Fontsize',14,'Location','northeast');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Susceptibles','Fontsize',16)
set(gca,'Fontsize',14)
title('(a)','Fontsize',16)

subplot(3,3,2)  % vaccinated, 1 dose
plot(t,V1PF+V1AZ,'-r','LineWidth',2), hold on
plot(t,V1AZ,'-b','LineWidth',2)
shade(t,V1PF+V1AZ,t,V1AZ,t,0*V1AZ,'FillType',[1 2;2 1;2 3],'FillColor',[1 0 0;0 1 0;0 0 1],'FillAlpha',[1 1 1]);
legend({'PZ','AZ'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Vaccinated (1 dose)','Fontsize',16)
set(gca,'Fontsize',14)
title('(b)','Fontsize',16)

subplot(3,3,3)  % vaccinated, both doses
plot(t,V2PF+V2AZ,'-r','LineWidth',2), hold on
plot(t,V2AZ,'-b','LineWidth',2)
shade(t,V2PF+V2AZ,t,V2AZ,t,0*V2AZ,'FillType',[1 2;2 1;2 3],'FillColor',[1 0 0;0 1 0;0 0 1],'FillAlpha',[1 1 1]);
legend({'PZ','AZ'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Vaccinated (2 doses)','Fontsize',16)
set(gca,'Fontsize',14)
title('(c)','Fontsize',16)

subplot(3,3,4)  % primary infections
plot(t,IP+IA+ID,'-r','LineWidth',2), hold on
plot(t,IA+ID,'-b','LineWidth',2)
plot(t,ID,'-g','LineWidth',2)
shade(t,IP+IA+ID,t,IA+ID,t,ID,t,0*ID,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Symptomatic infections (I)','Fontsize',16)
set(gca,'Fontsize',14)
title('(d)','Fontsize',16)

subplot(3,3,5)  % infections after vaccination
plot(t,IPV+IPR+IAV+IAR+IDV+IDR,'-r','LineWidth',2), hold on
plot(t,IAV+IAR+IDV+IDR,'-b','LineWidth',2)
plot(t,IDV+IDR,'-g','LineWidth',2)
shade(t,IPV+IPR+IAV+IAR+IDV+IDR,t,IAV+IAR+IDV+IDR,t,IDV+IDR,t,0*IDV,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Symptomatic infections (I_V+I_R)','Fontsize',16)
set(gca,'Fontsize',14)
title('(e)','Fontsize',16)

% subplot(3,3,6)  % infections after vaccination
% plot(t,IPR+IAR+IDR,'-r','LineWidth',2), hold on
% plot(t,IAR+IDR,'-b','LineWidth',2)
% plot(t,IDR,'-g','LineWidth',2)
% shade(t,IPR+IAR+IDR,t,IAR+IDR,t,IDR,t,0*IDR,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
% legend({'Wild-type','Alpha','Delta'},'Fontsize',14,'Location','best');
% axis tight
% xticks(dateticks);
% xticklabels(datelabels)
% ylabel('Symptomatic infections (R)','Fontsize',16)
% set(gca,'Fontsize',14)
% title('(f)','Fontsize',16)

subplot(3,3,6)  % asymtomatic infections
plot(t,AP+AA+AD,'-r','LineWidth',2), hold on
plot(t,AA+AD,'-b','LineWidth',2)
plot(t,AD,'-g','LineWidth',2)
shade(t,AP+AA+AD,t,AA+AD,t,AD,t,0*AD,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Asymptomatic infections (A)','Fontsize',16)
set(gca,'Fontsize',14)
title('(f)','Fontsize',16)

subplot(3,3,7)  % asymtomatic infections after vaccination/recovery
plot(t,APR+AAR+ADR,'-r','LineWidth',2), hold on
plot(t,AAR+ADR,'-b','LineWidth',2)
plot(t,ADR,'-g','LineWidth',2)
shade(t,APR+AAR+ADR,t,AAR+ADR,t,ADR,t,0*ADR,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Asymptomatic infections (A_R)','Fontsize',16)
set(gca,'Fontsize',14)
title('(g)','Fontsize',16)

subplot(3,3,8)  % asymtomatic infections
plot(t,RP+RA+RD,'-r','LineWidth',2), hold on
plot(t,RA+RD,'-b','LineWidth',2)
plot(t,RD,'-g','LineWidth',2)
shade(t,RP+RA+RD,t,RA+RD,t,RD,t,0*RD,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Fontsize',14,'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Recovered','Fontsize',16)
set(gca,'Fontsize',14)
title('(h)','Fontsize',16)

%%%%
figure(3), clf
subplot(2,2,1)
plot(tnew,movmean(dPnew+dAnew+dDnew,7),'-r','LineWidth',2); hold on
plot(tnew,movmean(dAnew+dDnew,7),'-b','LineWidth',2);
plot(tnew,movmean(dDnew,7),'-g','LineWidth',2);
shade(tnew,dPnew+dAnew+dDnew,tnew,dAnew+dDnew,tnew,dDnew,tnew,0*dDnew,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('New cases')
title('(a)')
set(gca,'Fontsize',18)

Icumtot = zeros(1,tn);
Acumtot = zeros(1,tn);
for i=2:tn
    Icumtot(i) = Icumtot(i-1) + (t(i)-t(i-1))*N*(dIPdt(i)+dIPVdt(i)+dIAdt(i)+dIAVdt(i)+dIDdt(i)+dIDVdt(i));
    Acumtot(i) = Acumtot(i-1) + (t(i)-t(i-1))*N*(dAPdt(i)+dAAdt(i)+dADdt(i));
end
% Icumtot = cumsum(N*(dIPdt+dIPVdt+dAPdt+dIAdt+dIAVdt));
% Acumtot = cumsum(N*(dAAdt+dIDdt+dIDVdt+dADdt));
subplot(2,2,2)
plot(t,Icumtot+Acumtot,'-r','LineWidth',2); hold on
plot(t,Acumtot,'-b','LineWidth',2);
shade(t,Icumtot+Acumtot,t,Acumtot,t,0*Acumtot,'FillType',[1 2;2 1;2 3],'FillColor',[1 0 0;0 1 0;0 0 1],'FillAlpha',[1 1 1]);
legend({'Symptomatic','Asymptomatic'},'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Total infections')
title('(b)')
set(gca,'Fontsize',18)

ii = min(find(t>daysact('1-jan-2020','27-jul-2021')));
[V1tot(ii)*100 V2tot(ii)*100]
subplot(2,2,3)
plot(t,V1tot*100,'-r','LineWidth',2); hold on
plot(t,V2tot*100,'-b','LineWidth',2);
shade(t,V1tot*100,t,V2tot*100,t,0*V2tot,'FillType',[1 2;2 1;2 3],'FillColor',[1 0 0;0 1 0;0 0 1],'FillAlpha',[1 1 1]);
legend({'One dose','Both doses'},'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Total vaccinations (%)')
title('(c)')
set(gca,'Fontsize',18)

Pdeathtot = N*DeadPtot;
Adeathtot = N*DeadAtot;
Ddeathtot = N*DeadDtot;
subplot(2,2,4)
plot(t,Pdeathtot+Adeathtot+Ddeathtot,'-r','LineWidth',2); hold on
plot(t,Adeathtot+Ddeathtot,'-b','LineWidth',2); 
plot(t,Adeathtot+Ddeathtot,'-g','LineWidth',2); 
shade(t,Pdeathtot+Adeathtot+Ddeathtot,t,Adeathtot+Ddeathtot,t,Ddeathtot,t,0*Ddeathtot,'FillType',[1 2;2 1;2 3;3 4],'FillColor',[1 0 0;0 1 0;0 0 1;0 1 0],'FillAlpha',[1 1 1 1]);
legend({'Wild-type','Alpha','Delta'},'Location','best');
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Total deaths')
title('(d)')
set(gca,'Fontsize',18)

tfin = 2*365;
beta_scale = zeros(1,tfin);
for i = 1:tfin
    beta_scale(i) = get_beta(i);
end

figure(4), clf
plot([1:tfin],beta_scale,'-k','LineWidth',2)
axis tight
xticks(dateticks);
xticklabels(datelabels)
ylabel('Lockdown effect (\lambda)');
ylim([0 1])
set(gca,'Fontsize',18)

