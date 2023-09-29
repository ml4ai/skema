% simulate lockdown
function [beta_scale] = get_beta(t)
ind = 1;

% first day of simulation
day0 = '1-jan-2020';

% state of emergency and such [2]
tlock(ind)   = daysact(day0,'23-mar-2020');
tunlock(ind) = daysact(day0,'17-jul-2020');
scale(ind)   = 0.4;
ind = ind+1;

% stage 3 [3]
tlock(ind)   = daysact(day0,'17-jul-2020');
tunlock(ind) = daysact(day0,'9-oct-2020');
scale(ind)   = 0.55;
ind = ind+1;

% oops, back to stage 2 [2]
tlock(ind)   = daysact(day0,'9-oct-2020');
tunlock(ind) = daysact(day0,'23-nov-2020');
scale(ind)   = 0.6;
ind = ind+1;

% strict lockdown, stay at home order [1]
tlock(ind)   = daysact(day0,'23-nov-2020');
% tunlock(ind) = daysact(day0,'5-mar-2021');
tunlock(ind) = daysact(day0,'5-jan-2021');
scale(ind)   = 0.55;
ind = ind+1;

% stricter lockdown, stay at home order [1]
tlock(ind)   = daysact(day0,'5-jan-2021');
tunlock(ind) = daysact(day0,'5-mar-2021');
scale(ind)   = 0.3; %0.25;
ind = ind+1;

% stay at home order lifted [2]
tlock(ind)   = daysact(day0,'5-mar-2021');
tunlock(ind) = daysact(day0,'13-apr-2021');
scale(ind)   = 0.4; %0.4;
ind = ind+1;

% lockdown again [1]
tlock(ind)   = daysact(day0,'13-apr-2021');
tunlock(ind) = daysact(day0,'11-jun-2021');
scale(ind)   = 0.15;% 0.1;
ind = ind+1;

% stage 2 reopening [3]
tlock(ind)   = daysact(day0,'11-jun-2021');
tunlock(ind) = daysact(day0,'2-jul-2021');
scale(ind)   = 0.2;%0.15;
ind = ind+1;

% stage 3 reopening [4]
tlock(ind)   = daysact(day0,'2-jul-2021');
tunlock(ind) = daysact(day0,'16-jul-2021');
scale(ind)   = 0.3;%0.25;
ind = ind+1;

% physical distancing [5]
tlock(ind)   = daysact(day0,'16-jul-2021');
tunlock(ind) = daysact(day0,'1-sep-2021');
scale(ind)   = 0.35;
ind = ind+1;

% physical distancing [5]
tlock(ind)   = daysact(day0,'1-sep-2021');
tunlock(ind) = daysact(day0,'1-nov-2021');
scale(ind)   = 0.4;
ind = ind+1;

% physical distancing loosening [6]
tlock(ind)   = daysact(day0,'1-nov-2021');
tunlock(ind) = daysact(day0,'1-jan-2022');
scale(ind)   = 0.5;
ind = ind+1;

% physical distancing tightened [6]
tlock(ind)   = daysact(day0,'1-jan-2022');
tunlock(ind) = daysact(day0,'1-jan-2023');
scale(ind)   = 0.4;
ind = ind+1;

ilock  = find(tlock<=t);
iunlock= find(t<tunlock);
ii     = intersect(ilock,iunlock);
if (length(ii)>1)
    [t,ii]
end
if (isempty(ii))
    beta_scale = 1;
else
    beta_scale = scale(ii);
end
