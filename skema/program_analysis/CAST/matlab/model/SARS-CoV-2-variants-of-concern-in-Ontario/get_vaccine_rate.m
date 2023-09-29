% returns vaccination rate and dose delay
function [vr1,vr2] = get_vaccine_rate(t)

day0    = '1-jan-2020';
t_vac0  = daysact(day0,'14-dec-2020');  % AZ actually started on Mar 3 2021
t_delay = daysact(day0,'31-may-2021');  % at first, dosing delay is 16 weeks
t_vac_peak = daysact(day0,'30-jun-2021');  % vaccination rate peaks at the end of June

% 55.2% Ontarian fully vaccinated, 68.7% at least one dose
fraction_first_dose = 0.6; % 0.558;
peak_vac_rate = 220*5e-5;%fraction_first_dose/14560;  % peak vaccination rate at the end of June
ss_vac_rate   = 10*5e-5;%fraction_first_dose/14560;  % long-term steady-state vaccination rate

% Pfizer vs AstraZeneca
eps_PZ = 0.95;
eps_AZ = 0.05;
% eps_PZ = 0.1;
% eps_AZ = 0.9;

if (t<t_vac0)  % no vaccine for the first year
    vr1 = [0;0];
    vr2 = [0;0];
elseif (t<t_delay)  % 16-week dose delay
    % assume vaccination rate linearly increases from 0 on 1/1/2021 to
    % 6/30/2021
    curr_vac_rate = peak_vac_rate * (t-t_vac0)/(t_vac_peak-t_vac0);
    % 95% Pfizer, 5% AZ
    vr1 = curr_vac_rate*[eps_PZ;eps_AZ];  
    vr2 = [1/(16*7);1/(16*7)];  % dosing delay, with some drop out
elseif (t<t_vac_peak)  % 3-week dose delay
    % assume vaccination rate linearly increases from 0 on 1/1/2021 to
    % 6/30/2021
    curr_vac_rate = peak_vac_rate * (t-t_vac0)/(t_vac_peak-t_vac0);
    % 95% Pfizer, 5% AZ
    vr1 = curr_vac_rate*[eps_PZ;eps_AZ];  
    vr2 = [1/28;1/(8*7)];  % dosing delay, with some drop out
else  % 3-week dose delay
    curr_vac_rate = (peak_vac_rate-ss_vac_rate) * exp(-3e-2*(t-t_vac_peak))+ss_vac_rate;
    % 95% Pfizer, 5% AZ
    vr1 = curr_vac_rate*[eps_PZ;eps_AZ]; 
    vr2 = [1/28;1/(8*7)];  % dosing delay, with some drop out
end
