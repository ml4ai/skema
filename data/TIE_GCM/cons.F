
program cons_module
!
! This software is part of the NCAR TIE-GCM.  Use is governed by the 
! Open Source Academic Research License Agreement contained in the file 
! tiegcmlicense.txt.
!
use params_module,only: dlat,dz,nlon,nlonp1,nlonp4,nlat,nlatp1,
|  dlev,nlev,nmlat,nmlon,nmlonp1,zmbot,zmtop,zibot,zitop,dlon,
|  glon,glat,glon1,glat1,zpmid,zpint,gmlon,gmlat,zpmag,zpimag,
|  dmlev,nlevp1,nmlevp1,nimlevp1,zpbot_dyn,zpibot_dyn
implicit none
!
! Define model constants. 
! Parameter constants are cons_module module data and are accessed 
!   in subroutines via use-association. 
! Derived constants are cons_module module data, and are calculated
!   in sub init_cons (contained in cons_module). 
!   Sub init_cons is called by sub init (init_mod.F).
! Parameter constants:
!
integer,parameter :: ndays =366  ! maximum number of days in a year
real,parameter ::
|  dzp  = dz,           ! alias for dz (also dlev)
|  re   = 6.37122e8,    ! earth radius (cm)                  C(51)
|  re_dyn = 6.378165e8, ! earth radius for apex
|  re_inv = 1./re,      ! inverse of earth radius            C(52)
|  avo  = 6.023e23,     ! avogadro number                    C(85)
|  boltz = 1.38E-16,    ! boltzman's constant                C(84)
|  p0   = 5.0e-4,       ! standard pressure                  C(81)
|  gask = 8.314e7,      ! gas constant                       C(57)
|  grav_par = 3.986004415e20 ! standard gravitational parameter (cm^3/sec^2)

!
! Smoothing constant is dependent on timestep (see below), so cannot be a parameter.
! (in previous revisions, shapiro=3.e-2, with no timestep or resolution dependence)
!
  real :: shapiro      ! shapiro smoother constant
  real :: default_step ! default timestep at current resolution.
  real :: smooth_fac   ! smoothing factor (typically 3.e-3)
!
integer :: nlonper=nlonp4 ! nlon + periodic points (alias for nlonp4)
!
! Many expressions require x/rmass, but its more efficient on some
! platforms to multiply rather than divide, so set rmassinv = 1./rmass 
! here, and use x*rmassinv in the code.
!
real,parameter :: 
|  rmass_o2 = 32., rmass_o1  = 16., rmass_n2  = 28., 
|  rmass_o3 = 48., rmass_n4s = 14., rmass_n2d = 14., 
|  rmass_no = 30., rmass_op  = 16., rmass_co2 = 44.,
|  rmass_he =  4., rmass_ar  = 40.
real,parameter ::
|  rmassinv_o2  = 1./rmass_o2,
|  rmassinv_o1  = 1./rmass_o1,
|  rmassinv_n2  = 1./rmass_n2,
|  rmassinv_o3  = 1./rmass_o3,
|  rmassinv_n4s = 1./rmass_n4s,
|  rmassinv_n2d = 1./rmass_n2d,
|  rmassinv_no  = 1./rmass_no,
|  rmassinv_op  = 1./rmass_op,
|  rmassinv_he  = 1./rmass_he,
|  rmassinv_ar  = 1./rmass_ar
!
! Model derived constants (see sub init_cons in this module):
!
real ::
|  pi,             ! set with 4*atan(1)    C(110)
|  sqrtpi,         ! sqrt(pi)
|  rtd,            ! radians-to-degrees (180./pi)
|  dtr,            ! degrees-to-radians (pi/180.)
|  dphi,           ! delta lat (pi/nlat)   C(2)
|  dphi_2div3,     ! 2./(3.*dphi)          C(12)
|  dphi_1div12,    ! 1./(12.*dphi)         C(13)
|  dphi2_5div2,    ! 5./(2.*dphi**2)
|  dphi2_4div3,    ! 4./(3.*dphi**2)
|  dphi2_1div12,   ! 1./(12.*dphi**2)
|  dlamda,         ! delta lon (2pi/nlon)  C(1)
|  dlamda_2div3,   ! 2./(3.*dlamda)        C(10)
|  dlamda_1div12,  ! 1./(12.*dlamda)       C(11)
|  dlamda2_5div2,  ! 5./(2.*dlamda**2)
|  dlamda2_4div3,  ! 4./(3.*dlamda**2)
|  dlamda2_1div12, ! 1./(12.*dlamda**2)
|  dt,             ! time step (secs)      C(4)
|  dtx2,           ! 2*dt                  C(6)
|  dtx2inv,        ! 1./(2*dt)             C(7)
|  freq_3m3,       ! frequency of 2-day wave (rad/sec)       C(21)
|  freq_semidi,    ! frequency of semidiurnal tide (rad/sec) C(23)
|  expzmid,        ! exp(-.5*dz)                             C(86)
|  expzmid_inv,    ! 1./expzmid                              C(87)
|  grav,           ! accel due to gravity (dependent on lower boundary)
|  dzgrav,         ! grav/gask C(65)
|  zbound          ! background low bound of Z (formerly ZBA in annual tide)
!
! Constants for dynamo and electric field calculations:
real,parameter :: h0 =9.0e6, r0 =re+h0    ! use mean earth radius
!
! Save gdlat,gdlon(nmlonp1,nmlat) in degrees from apex for sub define_mag (mpi.F)
!     real,dimension(nmlonp1,nmlat) :: gdlatdeg,gdlondeg

real,parameter :: hs=1.3e7
real ::
|  dlatg, dlong, dlatm, dlonm,dmagphrlon

!
! Special pi for mag field calculations. If pi=4.*atan(1.) and code is
! linked with -lmass lib, then the last 2 digits (16th and 17th) of pi
! are different (56 instead of 12), resulting in theta0(j=49)==0., which 
! is wrong (should be .1110e-15).
!
real,parameter :: pi_dyn=3.14159265358979312

integer,parameter :: difhor=1

end program cons_module
