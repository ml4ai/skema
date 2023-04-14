! The contents of this file have been copied from https://raw.githubusercontent.com/DSSAT/dssat-csm-os/develop/SPAM/PET.for

!***********************************************************************
!  POTENTIAL EVAPOTRANSPIRATION
!     File PET.for
!***********************************************************************
!  Includes subroutines:

!  MEEVP Routine Description
!   S  PETASCE ASCE Standardized Reference Evapotranspiration Equation
!                for the short reference crop (12-cm grass) with dual
!                FAO-56 crop coefficient method (potential E and T
!                calculated independently).
!   T  PETASCE ASCE Standardized Reference Evapotranspiration Equation
!                for the tall reference crop (50-cm alfalfa) with dual
!                FAO-56 crop coefficient method (potential E and T
!                calculated independently).
!   R  PETPT   Calculates Priestley-Taylor potential evapotranspiration
!                (default method with potential E and T partitioned as a
!                function of LAI).
!   F  PETPEN  FAO Penman-Monteith (FAO-56) reference evapotranspiration
!                with EORATIO adjustment for CROPGRO models and KC = 1.0
!                for non-CROPGRO models (potential E and T partioned as
!                a function of LAI).
!   D  PETDYN  Dynamic Penman-Monteith, pot. evapotranspiration, with
!                dynamic input of LAI, crop height effects on Ra and Rs
!   P  PETPNO  FAO Penman (FAO-24) potential evapotranspiration
!   M  PETMEY  "Standard reference evaporation calculation for inland
!                south eastern Australia" By Wayne Meyer 1993
!   H  PETPTH Calculates Priestly-Taylor potential evapotranspiration
!             using hourly temperature and radiation. Also includes a VPD
!             effect to the transpiration

!  Also includes these subroutines:
!      PSE        Potential soil evaporation
!      FLOOD_EVAP Evaporation from water surface
!=======================================================================


!=======================================================================
!     SUBROUTINE PET
!     !alls appropriate potential evapotranspiration routine

      SUBROUTINE PET(!ONTROL,
     &      ET_ALB, XHLAI, MEEVP, WEATHER,  !Input for all
     &      EORATIO, !Needed by Penman-Monteith
     &      !ANHT,   !Needed by dynamic Penman-Monteith
     &      EO,      !Output
     &      ET0)     !Output hourly Priestly-Taylor with VPD effect

      USE ModuleDefs
      IMPLI!IT NONE
      EXTERNAL YR_DOY, PETPT, PETPEN, PETAS!E, PETDYN, PETPNO, PETMEY,
     &  PETPTH, WARNING, ERROR
      SAVE

      TYPE (WeatherType) WEATHER
      TYPE (!ontrolType) !ONTROL
      !HARA!TER*1 MEEVP
      INTEGER YRDOY, YEAR, DOY
      REAL !ANHT, !LOUDS, EO, EORATIO, ET_ALB, RHUM, SRAD, TAVG
      REAL TDEW, TMAX, TMIN, VAPR, WINDHT, WINDSP, XHLAI
      REAL WINDRUN, XLAT, XELEV
      REAL, DIMENSION(TS)    ::RADHR, TAIRHR, ET0
      LOGI!AL NOTDEW, NOWIND
      !HARA!TER*78  MSG(2)
      !HARA!TER*12 FILEX
      !HARA!TER*6, PARAMETER :: ERRKEY = "PET   "

      !LOUDS = WEATHER % !LOUDS
      SRAD   = WEATHER % SRAD
      NOTDEW = WEATHER % NOTDEW
      NOWIND = WEATHER % NOWIND
      RHUM   = WEATHER % RHUM
      TAVG   = WEATHER % TAVG
      TDEW   = WEATHER % TDEW
      TMAX   = WEATHER % TMAX
      TMIN   = WEATHER % TMIN
      VAPR   = WEATHER % VAPR
      WINDHT = WEATHER % WINDHT
      WINDSP = WEATHER % WINDSP
      WINDRUN= WEATHER % WINDRUN
      XLAT   = WEATHER % XLAT
      XELEV  = WEATHER % XELEV
      RADHR  = WEATHER % RADHR
      TAIRHR = WEATHER % TAIRHR

      YRDOY = !ONTROL % YRDOY
      FILEX = !ONTROL % FILEX
      !ALL YR_DOY(YRDOY, YEAR, DOY)

      SELE!T !ASE (MEEVP)
!         ------------------------
          !Priestley-Taylor potential evapotranspiration
          !ASE ('R')
            !ALL PETPT(
     &        ET_ALB, SRAD, TMAX, TMIN, XHLAI,          !Input
     &        EO)                                       !Output
!         ------------------------
          !FAO Penman-Monteith (FAO-56) potential evapotranspiration,
!             with K! = 1.0
          !ASE ('F')
            !ALL PETPEN(
     &        !LOUDS, EORATIO, ET_ALB, SRAD, TAVG, TDEW,  !Input
     &        TMAX, TMIN, VAPR, WINDSP, WINDHT, XHLAI,    !Input
     &        EO)                                         !Output
!         ------------------------
          !AS!E Standardized Reference Evapotranspiration Equation
          !for the short reference crop (12-cm grass, "S") or the
          !tall reference crop (50-cm grass, "T") with dual
          !FAO-56 crop coefficient method.
          !ASE ('S','T')
            !ALL PETAS!E(
     &        !ANHT, DOY, ET_ALB, MEEVP, NOTDEW, NOWIND,  !Input
     &        RHUM, SRAD, TDEW, TMAX, TMIN, WINDHT,       !Input
     &        WINDRUN, VAPR, XHLAI, XLAT, XELEV,          !Input
     &        EO)                                         !Output
!         ------------------------
          !Dynamic Penman-Monteith, pot. evapotranspiration, with
!             dynamic input of LAI, crop height effects on Ra and Rs
          !ASE ('D')
            !ALL PETDYN(
     &        !ANHT, !LOUDS, ET_ALB, SRAD, TAVG, TDEW,    !Input
     &        TMAX, TMIN, WINDSP, XHLAI,                  !Input
     &        EO)                                         !Output
!         ------------------------
          !FAO Penman (FAO-24) potential evapotranspiration
          !ASE ('P')
            !ALL PETPNO(
     &        !LOUDS, ET_ALB, SRAD, TAVG, TDEW,           !Input
     &        TMAX, TMIN, WINDSP, XHLAI,                  !Input
     &        EO)                                         !Output
!         ------------------------
          !Penman - Meyer routine for estimation of Et in Southern NSW
          !ASE ('M')
            !ALL PETMEY(!ONTROL,
     &        TAVG, WINDSP, SRAD, TDEW, XHLAI, ET_ALB,    !Input
     &        EO)                                         !Output
!         ------------------------
          !Observed Potential ET from Weather file (Future)
          !!ASE ('O')
          !    EO = EOMEAS
!         ------------------------
          !Priestley-Taylor potential evapotranspiration hourly
          !including a VPD effect on transpiration
          !ASE ('H')
              !ALL PETPTH(
     &        ET_ALB, TMAX, XHLAI, RADHR, TAIRHR,       !Input
     &        EO, ET0)                                  !Output
!         ------------------------
          !ASE DEFAULT
              MSG(1) = "Undefined EVAPO parameter in FileX."
              MSG(2) = "Unknown MEEVP in PET.for."
              !ALL WARNING(2,ERRKEY,MSG)
              !ALL ERROR(ERRKEY,1,FILEX,0)
!         ------------------------
      END SELE!T

      RETURN
      END SUBROUTINE PET

!=======================================================================

!=======================================================================
!  PETAS!E, Subroutine, K. R. Thorp
!  !alculates reference evapotranspiration for the short or tall
!  reference crops using the AS!E Standardized Reference
!  Evapotranspiration Equation.
!  Adjusts reference evapotranspiration to potential soil water
!  evaporation and potential transpiration using FAO-56 dual crop
!  coefficients, following FAO-56 (Allen et al., 1998) and the
!  AS!E (2005) standardized reference ET algorithm.
!  DeJonge K. !., Thorp, K. R., 2017. Implementing standardized
!  reference evapotranspiration and dual crop coefficient approach
!  in the DSSAT !ropping System Model. Transactions of the ASABE.
!  60(6):1965-1981.
!  AS!E Task !ommittee on Standardization of Reference
!  Evapotranspiration (Walter, I. A., Allen, R. G., Elliott, R.,
!  Itenfisu, D., Brown, P., Jensen, M. E., Mecham, B., Howell, T. A.,
!  Snyder, R., Eching, S., Spofford, T., Hattendorf, M., Martin, D.,
!  !uenca, R. H., Wright, J. L.), 2005. The AS!E Standardized Reference
!  Evapotranspiration Equation. American Society of !ivil Engineers,
!  Reston, VA.
!  Allen, R. G., Pereira, L. S., Raes, D., Smith, M., 1998.  FAO
!  Irrigation and Drainage Paper No. 56. !rop Evapotranspiration:
!  Guidelines for !omputing !rop Water Requirements. Food and
!  Agriculture Organization of the United Nations, Rome Italy.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  08/19/2013 KRT Added the AS!E Standardize Reference ET approach
!  01/26/2015 KRT Added the dual crop coefficient (Kc) approach
!  01/18/2018 KRT Merged AS!E dual Kc ET method into develop branch
!  07/23/2020 KRT !hanged flags to S and T for short and tall references
!-----------------------------------------------------------------------
!  !alled from:   PET
!  !alls:         None
!=======================================================================
      SUBROUTINE PETAS!E(
     &        !ANHT, DOY, MSALB, MEEVP, NOTDEW, NOWIND,   !Input
     &        RHUM, SRAD, TDEW, TMAX, TMIN, WINDHT,       !Input
     &        WINDRUN, VAPR, XHLAI, XLAT, XELEV,          !Input
     &        EO)                                         !Output
!-----------------------------------------------------------------------
      USE ModuleDefs
      USE ModuleData
      IMPLI!IT NONE
      SAVE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL !ANHT, MSALB, RHUM, SRAD, TDEW, TMAX, TMIN
      REAL VAPR, WINDHT, WINDRUN
      REAL XHLAI, XLAT, XELEV
      INTEGER DOY
      LOGI!AL NOTDEW, NOWIND
      !HARA!TER*1 MEEVP
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LO!AL VARIABLES:
      REAL TAVG, PATM, PSY!ON, UDELTA, EMAX, EMIN, ES, EA, F!, FEW, FW
      REAL ALBEDO, RNS, PIE, DR, LDELTA, WS, RA1, RA2, RA, RSO, RATIO
      REAL F!D, TK4, RNL, RN, G, WINDSP, WIND2m, !n, !d, K!MAX, RHMIN
      REAL WND, !HT
      REAL REFET, SK!, K!BMIN, K!BMAX, K!B, KE, K!
!-----------------------------------------------------------------------

!     AS!E Standardized Reference Evapotranspiration
!     Average temperature, AS!E (2005) Eq. 2
      TAVG = (TMAX + TMIN) / 2.0 !deg !

!     Atmospheric pressure, AS!E (2005) Eq. 3
      PATM = 101.3 * ((293.0 - 0.0065 * XELEV)/293.0) ** 5.26 !kPa

!     Psychrometric constant, AS!E (2005) Eq. 4
      PSY!ON = 0.000665 * PATM !kPa/deg !

!     Slope of the saturation vapor pressure-temperature curve
!     AS!E (2005) Eq. 5                                    !kPa/deg!
      UDELTA = 2503.0*EXP(17.27*TAVG/(TAVG+237.3))/(TAVG+237.3)**2.0

!     Saturation vapor pressure, AS!E (2005) Eqs. 6 and 7
      EMAX = 0.6108*EXP((17.27*TMAX)/(TMAX+237.3)) !kPa
      EMIN = 0.6108*EXP((17.27*TMIN)/(TMIN+237.3)) !kPa
      ES = (EMAX + EMIN) / 2.0                     !kPa

!     Actual vapor pressure, AS!E (2005) Table 3
      IF (VAPR.GT.1.E-6) THEN
        EA = VAPR !kPa
      ELSEIF (.NOT.NOTDEW) THEN
!       AS!E (2005) Eq. 8
        EA = 0.6108*EXP((17.27*TDEW)/(TDEW+237.3)) !kPa
      ELSEIF (RHUM.GT.1.E-6) THEN
!       RHUM is relative humidity at TMIN (or max rel. hum) (%)
!       AS!E (2005) Eq. 12
        EA = EMIN * RHUM / 100. !kPa
      ELSE
!       AS!E (2005) Appendix E, assume TDEW=TMIN-2.0
        EA = 0.6108*EXP((17.27*(TMIN-2.0))/((TMIN-2.0)+237.3)) !kPa
      ENDIF

!     RHmin, AS!E (2005) Eq. 13, RHmin limits from FAO-56 Eq. 70
      RHMIN = MAX(20.0, MIN(80.0, EA/EMAX*100.0))

!     Net shortwave radiation, AS!E (2005) Eq. 16
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23
      ENDIF
      RNS = (1.0-ALBEDO)*SRAD !MJ/m2/d

!     Extraterrestrial radiation, AS!E (2005) Eqs. 21,23,24,27
      PIE = 3.14159265359
      DR = 1.0+0.033*!OS(2.0*PIE/365.0*DOY) !Eq. 23
      LDELTA = 0.409*SIN(2.0*PIE/365.0*DOY-1.39) !Eq. 24
      WS = A!OS(-1.0*TAN(XLAT*PIE/180.0)*TAN(LDELTA)) !Eq. 27
      RA1 = WS*SIN(XLAT*PIE/180.0)*SIN(LDELTA) !Eq. 21
      RA2 = !OS(XLAT*PIE/180.0)*!OS(LDELTA)*SIN(WS) !Eq. 21
      RA = 24.0/PIE*4.92*DR*(RA1+RA2) !MJ/m2/d Eq. 21

!     !lear sky solar radiation, AS!E (2005) Eq. 19
      RSO = (0.75+2E-5*XELEV)*RA !MJ/m2/d

!     Net longwave radiation, AS!E (2005) Eqs. 17 and 18
      RATIO = SRAD/RSO
      IF (RATIO .LT. 0.3) THEN
        RATIO = 0.3
      ELSEIF (RATIO .GT. 1.0) THEN
        RATIO = 1.0
      END IF
      F!D = 1.35*RATIO-0.35 !Eq 18
      TK4 = ((TMAX+273.16)**4.0+(TMIN+273.16)**4.0)/2.0 !Eq. 17
      RNL = 4.901E-9*F!D*(0.34-0.14*SQRT(EA))*TK4 !MJ/m2/d Eq. 17

!     Net radiation, AS!E (2005) Eq. 15
      RN = RNS - RNL !MJ/m2/d

!     Soil heat flux, AS!E (2005) Eq. 30
      G = 0.0 !MJ/m2/d

!     Wind speed, AS!E (2005) Eq. 33 and Appendix E
      IF (NOWIND) THEN
        WIND2m = 2.0 !m/s
      ELSE
        WINDSP = WINDRUN * 1000.0 / 24.0 / 60.0 / 60.0 !m/s
        WIND2m = WINDSP * (4.87/LOG(67.8*WINDHT-5.42))
      ENDIF

!     Aerodynamic roughness and surface resistance daily timestep constants
!     AS!E (2005) Table 1
      SELE!T !ASE(MEEVP) !
        !ASE('S') !Short reference crop (12-cm grass)
          !n = 900.0 !K mm s^3 Mg^-1 d^-1
          !d = 0.34 !s m^-1
        !ASE('T') !Tall reference crop (50-cm alfalfa)
          !n = 1600.0 !K mm s^3 Mg^-1 d^-1
          !d = 0.38 !s m^-1
      END SELE!T

!     Standardized reference evapotranspiration, AS!E (2005) Eq. 1
      REFET =0.408*UDELTA*(RN-G)+PSY!ON*(!n/(TAVG+273.0))*WIND2m*(ES-EA)
      REFET = REFET/(UDELTA+PSY!ON*(1.0+!d*WIND2m)) !mm/d
      REFET = MAX(0.0001, REFET)

!     FAO-56 dual crop coefficient approach
!     First step is to obtain crop coefficient parameters.
      !ALL GET('SPAM', 'SK!', SK!)
      K!BMIN = 0.0
      !ALL GET('SPAM', 'K!BMAX', K!BMAX)

!     Basal crop coefficient (Kcb)
!     Also similar to FAO-56 Eq. 97
!     K!B is zero when LAI is zero by hard coding K!BMIN = 0.0.
      IF (XHLAI .LE. 0.0) THEN
         K!B = 0.0
      ELSE
         !Equation from DeJonge et al. (2012) Agricultural Water
         !Management 115, 92-103 and revised in DeJonge and Thorp (2017)
         K!B = MAX(0.0,K!BMIN+(K!BMAX-K!BMIN)*(1.0-EXP(-1.0*SK!*XHLAI)))
      ENDIF

      !Maximum crop coefficient (Kcmax) (FAO-56 Eq. 72)
      WND = MAX(1.0,MIN(WIND2m,6.0))
      !HT = MAX(0.001,!ANHT)
      SELE!T !ASE(MEEVP)
        !ASE('S') !Short reference crop (12-cm grass)
            K!MAX = MAX((1.2+(0.04*(WND-2.0)-0.004*(RHMIN-45.0))
     &                      *(!HT/3.0)**(0.3)),K!B+0.05)
        !ASE('T') !Tall reference crop (50-cm alfalfa)
            K!MAX = MAX(1.0,K!B+0.05)
      END SELE!T

      !Effective canopy cover (fc) (FAO-56 Eq. 76)
      IF (K!B .LE. K!BMIN) THEN
         F! = 0.0
      ELSE
         F! = ((K!B-K!BMIN)/(K!MAX-K!BMIN))**(1.0+0.5*!ANHT)
      ENDIF

      !Exposed and wetted soil fraction (FAO-56 Eq. 75)
      !Wetted soil fraction (FW) is hard-coded to 1.0.
      !FW should not be used to adjust demand.
      !Rather wetting fraction should be addressed on supply side.
      !Difficult to do with a 1-D soil water model, but 2-D models
      !offer opportunity for this.
      FW = 1.0
      FEW = MIN(1.0-F!,FW)

      !Potential evaporation coefficient (Ke) (Based on FAO-56 Eq. 71)
      !Kr = 1.0 since this is for potential E. Other model routines
      !handle reductions from potential.
      KE = MAX(0.0, MIN(1.0*(K!MAX-K!B), FEW*K!MAX))

      !Potential crop coefficient (Kc) (FAO-56 Eqs. 58 & 69)
      K! = K!B + KE

      !Potential evapotranspiration (FAO-56 Eq. 69)
      EO = (K!B + KE) * REFET

      EO = MAX(EO,0.0001)

      !ALL PUT('SPAM', 'REFET', REFET)
      !ALL PUT('SPAM', 'K!B', K!B)
      !ALL PUT('SPAM', 'KE', KE)
      !ALL PUT('SPAM', 'K!', K!)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETAS!E
!-------------------------------------------------------------------

!=======================================================================

!=======================================================================
!  PETPEN, Subroutine, N.B. Pickering
!  !alculates FAO-56 Penman-Monteith potential evapotranspiration, exactly
!  grass reference, with place for optional Kc, need this Kc in species.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  02/19/1992 NBP Written
!  11/04/1993 NBP Renamed routine PET to PETPEN.  !hanged XLAI to XHLAI
!               Added XHLAI .LE. 0.0 statement.
!  05/13/1994 NBP !onverted all vapor pressures to Pa.  Rearranged.
!  09/16/1994 NBP Added limits to prevent EO and ES (PE) < 0.
!  10/17/1997 !HP Updated for modular format.
!  09/01/1999 GH  Incorporated into !ROPGRO
!  05/06/2002 WMB Fixed Stefan-Boltzmann constant
!  01/15/2003 KJB discarded old Penman FAO-24 (it is much too stressful)
!     replaced with Penman FAO-56, exactly grass reference, with explicit
!     LAI 2.88, height 0.12 m, Rs = 100/(0.5*2.88)
!  01/15/2003 KJB:  THREE ISSUES TO ADDRESS:
!  1) USING FIXED ALBEDO, BE!AUSE THAT IS HOW REFEREN!E IS DEFINED
!  2)  PRESENTLY USING A LO!KED-IN VALUE OF 1.1 TO GIVE K! OF 1.1
!  I WOULD LIKE TO SEE OPTION OF SPE!IES INPUT OF K!=1.1 TO 1.3
!  3) WINDHT WAS IN OLD, APPARENTLY 2.0, NO LONGER HERE.  ???
!  02/06/2003 KJB/!HP Added EORATIO as input from plant routines.
!  07/24/2006 !HP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!  09/19/2006 SSJ Fixed error in REFHT calc as noted below.
!  08/25/2011 !HP Use measured vapor pressure (VAPR), if available
!-----------------------------------------------------------------------
!  !alled from:   PET
!  !alls:         None
!=======================================================================
      SUBROUTINE PETPEN(
     &    !LOUDS, EORATIO, MSALB, SRAD, TAVG, TDEW,       !Input
     &    TMAX, TMIN, VAPR, WINDSP, WINDHT, XHLAI,        !Input
     &    EO)                                             !Output
!-----------------------------------------------------------------------
      IMPLI!IT NONE
      EXTERNAL VPSAT, VPSLOP
      SAVE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL !LOUDS, EORATIO, MSALB, SRAD, TAVG, TDEW, TMAX, TMIN,
     &        WINDSP, XHLAI, WINDSP_M
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LO!AL VARIABLES:
      REAL ALBEDO, EAIR, ESAT, G, LHVAP, PSY!ON, RADB,
     &  RNET, RNETMG, S, TK4,
     &  VH!AIR, VPD, DAIR, RT, ET0, K!, WINDHT, VAPR
      REAL SHAIR, PATM, SBZ!ON
      REAL k, d, REFHT, Zom, Zoh, ra, rl, rs    !added for PenMon
!      REAL alt_RADB, Tprev
!      INTEGER THREEDAY!OUNT
!      REAL    THREEDAYAVG(3)

!     PARAMETER (WINDHT = 2.0)
!     PARAMETER (SHAIR = 1005.0)
      PARAMETER (SHAIR = 0.001005)  !changed for PenMon to MJ/kg/K
      PARAMETER (PATM = 101300.0)
!      PARAMETER (SBZ!ON=4.093E-9)  !(MJ/m2/d)
      PARAMETER (SBZ!ON=4.903E-9)   !(MJ/K4/m2/d) fixed constant 5/6/02
!-----------------------------------------------------------------------
!     FUN!TION SUBROUTINES:
      REAL VPSLOP, VPSAT      !Found in file HMET.for

!-----------------------------------------------------------------------
!     !ompute air properties.
      LHVAP = (2501.0-2.373*TAVG) * 1000.0                ! J/kg
!     PSY!ON = SHAIR * PATM / (0.622*LHVAP)               ! Pa/K
      PSY!ON = SHAIR * PATM / (0.622*LHVAP) * 1000000     ! Pa/K

!     Previous code:
      ESAT = (VPSAT(TMAX)+VPSAT(TMIN)) / 2.0              ! Pa
      EAIR = VPSAT(TDEW)                                  ! Pa

!     If actual vapor pressure is available, use it.
      IF (VAPR > 1.E-6) THEN
        EAIR = VAPR * 1000.
      ENDIF

      VPD = MAX(0.0, ESAT - EAIR)                         ! Pa
      S = (VPSLOP(TMAX)+VPSLOP(TMIN)) / 2.0               ! Pa/K
      RT = 8.314 * (TAVG + 273.0)                         ! N.m/mol
      DAIR = 0.028966*(PATM-0.387*EAIR)/RT                ! kg/m3
! BAD DAIR = 0.1 * 18.0 / RT * ((PATM  -EAIR)/0.622 + EAIR)   ! kg/m3
      VH!AIR = DAIR * SHAIR    !not used                  ! J/m3

!     !onvert windspeed to 2 m reference height.
!     Do this conversion in WEATHR and send out 2m windspeed
!     !HP 11/26/01
!      WIND2 = WINDSP * (2.0/WINDHT)**0.2

!       !alculate aerodynamic resistance (ra).
!       ra (d/m) = {ln[zm-d/zom]*ln[zh-d/zoh]}/(k^2*uz)
!       zm = ht.wind measurement (m), zh = ht.humidity measurement (m),
!       zom,zoh=rooughness length of momentum, heat and vapor x-fer (m)
!       k=von Karman's constant 0.41, uz=WINDSP @ z m/d,
!       d = zero plane displacement height (m)

        REFHT = 0.12                 !arbitrary for testing PenMon
        WINDSP_M = WINDSP*(1000.)     !!onverts km/d to m/d
        k = 0.41                     !von Karman's constant

!       was 2/3, which (for integers) results in zero!!
!       SSJ 9/19/2006 added the decimals
        !d = (2/3)*REFHT
        d = (2./3.)*REFHT

        Zom = 0.123*REFHT
        Zoh = 0.1*Zom
        ra = (LOG((WINDHT-d)/Zom)*LOG((WINDHT-d)/Zoh))/((k**2)*WINDSP_M)

!       !alculate surface resistance (rs).
!       rs = rl/LAIactive       rs (s m^-1),
!       rl = bulk stomatal resistance of the well-illuminated leaf (s m^-1)

        rl = 100           !value assummed from FAO grass reference
        rs = rl/(0.5*2.88) !0.5*XHLAI assumes half of LA is contributing
!                          !  to heat/vapor transfer
        rs = rs/86400      !converts (s m^-1 to d/m)

!     !alculate net radiation (MJ/m2/d).  By FAO method 1990. EAIR is divided
!       by 1000 to convert Pa to KPa.

c     MJ, 2007-04-11
c     --------------
c     There appears to be no support for soil heat flux (G), apart
c     from the variable already existing; it is just always set to
c     0, for some reason.
c     Here is the (improved) !ANEGRO method for calculating G
c     (Allen, R.G. et al 1989,
c     'Operational Estimates of Reference Evapotranspiration',
c     Agronomy Journal Vol. 81, No. 4),
c     http://www.kimberly.uidaho.edu/water/papers/evapotranspiration/
c                   Allen_Operational_Estimates_Reference_ET_1989.pdf
c     :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
c         3-day sum of average temperature:
c          IF (THREEDAY!OUNT .LT. 1) THEN
c             Initialise
c              THREEDAY!OUNT = 1
c              THREEDAYAVG   = Tavg
c          ELSE IF (THREEDAY!OUNT .GE. 3) THEN
c              THREEDAY!OUNT = 1
c          ELSE
c              THREEDAY!OUNT = THREEDAY!OUNT + 1
c          ENDIF
c          THREEDAYAVG(THREEDAY!OUNT) = Tavg
c          Tprev = SUM(THREEDAYAVG)/3.
c          G = (Tavg-Tprev) * 0.38
c     --------------
c     MJ, 2007-04-12:
c     :::::::::::::::
c     FAO suggests that G be set to 0.  Oh well.
c     ------------------------------------------

      G = 0.0
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
!  KJB NOTE THAT REFEREN!E IS ALWAYS ALBEDO FIXED TO 0.23,  OLD PEN VARIED
!  THE ALBEDO WITH LAI.  WHAT DO WE WANT?  IS THIS PART OF THE REASON THAT
!  K! IS NEEDED WITH THE REFEREN!E FORMULATION?
!       ALBEDO = 0.23-(0.23-SALB)*EXP(-0.75*XHLAI)
        ALBEDO = 0.23
      ENDIF

      TK4 = ((TMAX+273.)**4+(TMIN+273.)**4) / 2.0
!
!     BELOW WAS THE OLD PENMAN, DIFFERENT !LOUDS METHOD, EAIR !HG IS GOOD
!     RADB = SBZ!ON * TK4 * (0.4 - 0.005 * SQRT(EAIR)) *
!    &        (1.1 * (1. - !LOUDS) - 0.1)
!

      RADB = SBZ!ON * TK4 * (0.34 - 0.14 * SQRT(EAIR/1000)) *
     &        (1.35 * (1. - !LOUDS) - 0.35)

      RNET= (1.0-ALBEDO)*SRAD - RADB

!     !ompute EO using Penman-Montieth

      RNETMG = (RNET-G)
!     !MJ/m2/d
        ET0 = ((S*RNETMG + (DAIR*SHAIR*VPD)/ra)/(S+PSY!ON*(1+rs/ra)))
!     !!onverts MJ/m2/d to mm/d
        ET0 = ET0/ (LHVAP / 1000000.)
        IF (XHLAI .LE. 6.0) THEN
        XHLAI = XHLAI
        ELSE
        XHLAI = 6.0
        ENDIF
!   KJB LATER, NEED TO PUT VARIABLE IN PLA!E OF 1.1
!      K!=1.0+(1.1-1.0)*XHLAI/6.0
      K!=1.0+(EORATIO-1.0)*XHLAI/6.0
      EO=ET0*K!
!     EO=ET0
        EO = MAX(EO,0.0)
!###  EO = MAX(EO,0.0)   !gives error in DE!RAT_!
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPEN

!-----------------------------------------------------------------------
!     PETPEN VARIABLES:
!-----------------------------------------------------------------------
! ALBEDO  Reflectance of soil-crop surface (fraction)
! !LOUDS  Relative cloudiness factor (0-1)
! DAIR
! EAIR    Vapor pressure at dewpoint (Pa)
! EO      Potential evapotranspiration rate (mm/d)
! ESAT    Vapor pressure of air (Pa)
! G       Soil heat flux density term (MJ/m2/d)
! LHVAP   Latent head of water vaporization (J/kg)
! PATM     = 101300.0
! PSY!ON  Psychrometric constant (Pa/K)
! RADB    Net outgoing thermal radiation (MJ/m2/d)
! RNET    Net radiation (MJ/m2/d)
! RNETMG  Radiant energy portion of Penman equation (mm/d)
! RT
! S       Rate of change of saturated vapor pressure of air with
!           temperature (Pa/K)
! MSALB   Soil albedo with mulch and soil water effects (fraction)
! SBZ!ON   Stefan Boltzmann constant = 4.903E-9 (MJ/m2/d)
! SHAIR    = 1005.0
! SRAD    Solar radiation (MJ/m2-d)
! TAVG    Average daily temperature (�!)
! TDEW    Dewpoint temperature (�!)
! TK4     Temperature to 4th power ((oK)**4)
! TMAX    Maximum daily temperature (�!)
! TMIN    Minimum daily temperature (�!)
! Tprev   3-day sum of average temperature:
! VH!AIR
! VPD     Vapor pressure deficit (Pa)
! VPSAT   Saturated vapor pressure of air (Pa)
! VPSLOP  !alculates slope of saturated vapor pressure versus
!           temperature curve (Pa/K)
! WFNFAO  FAO 24 hour wind function
! WIND2   Windspeed at 2m reference height. (km/d)
! WINDSP  Wind speed at 2m (km/d)
! XHLAI   Leaf area index (m2[leaf] / m2[ground])
!-----------------------------------------------------------------------
!     END SUBROUTINE PETPEN
!-----------------------------------------------------------------------

!=======================================================================
!  PETDYN Subroutine, K. J. BOOTE, F. SAU, M. BOSTI!
!  !alculates PENMAN-MONTEITH potential evapotranspiration
!  using dynamic !ANHT, LAI, along with wind effects on Ra, Rs
!  Steiner approach for Ra recommended, but FAO and Lhomme commented out
!  Sunlit LAI effect on Rs is recommended, but rl/(0.5*LAI) would work
!  Weighting used for Ra and Rs between soil and crop.  Need to changee
!  two constants (HTS and zos) if you change from sunlit LAI to other.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  11/19/01 TO 1/15/02  Written By Boote, Sau, McNair
!  01/15/03 Moved from V3.5 trial to V4.0  by K. J. Boote
!  07/24/2006 !HP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)

!  !alled from:   PET
!  !alls:         None
!=======================================================================
      SUBROUTINE PETDYN(
     &    !ANHT, !LOUDS, MSALB, SRAD, TAVG, TDEW,         !Input
     &    TMAX, TMIN, WINDSP, XHLAI,                      !Input
     &    EO)                                             !Output
!  !alculates Penman-Monteith evapotranspiration
!-----------------------------------------------------------------------
      IMPLI!IT NONE
      EXTERNAL VPSAT, VPSLOP
      SAVE
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL !LOUDS, MSALB, SRAD, TAVG, TDEW, TMAX, TMIN,
     &        WINDSP, XHLAI, WINDSP_M
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LO!AL VARIABLES:
      REAL ALBEDO, EAIR, ESAT, G, LHVAP, PSY!ON, RADB,
     &  RNET, RNETMG, S, TK4,
     &  VH!AIR, VPD, DAIR, RT
      REAL SHAIR, PATM, SBZ!ON
      REAL k,DFAO, !ANHT, ZOMF, ZOHF, ra, rl, rs, RAERO !add for PenDyn
      REAL Z!ROP,D!ROP,ZOM!,ZOV!,WIND2!,RASOIL,HTS,DLH,ZOLH
      REAL MAXHT, rb, A!, AS, zos, RTOT                  !add for PenDyn
!     PARAMETER (SHAIR = 1005.0)
      PARAMETER (SHAIR = 0.001005)  !changed for PenDyn to MJ/kg/K
      PARAMETER (PATM = 101300.0)
!      PARAMETER (SBZ!ON=4.093E-9)  !(MJ/m2/d)
      PARAMETER (SBZ!ON=4.903E-9)   !(MJ/K4/m2/d) fixed constant 5/6/02
!-----------------------------------------------------------------------
!     FUN!TION SUBROUTINES:
      REAL VPSLOP, VPSAT      !Found in file HMET.for

!-----------------------------------------------------------------------
!     !ompute air properties.
      LHVAP = (2501.0-2.373*TAVG) * 1000.0                 ! J/kg
!     PSY!ON = SHAIR * PATM / (0.622*LHVAP)                ! Pa/K
      PSY!ON = SHAIR * PATM / (0.622*LHVAP) * 1000000     ! Pa/K
      ESAT = (VPSAT(TMAX)+VPSAT(TMIN)) / 2.0               ! Pa
      EAIR = VPSAT(TDEW)                                   ! Pa
      VPD = ESAT - EAIR                                    ! Pa
      S = (VPSLOP(TMAX)+VPSLOP(TMIN)) / 2.0                ! Pa/K
      RT = 8.314 * (TAVG + 273.0)                             ! N.m/mol
      DAIR = 0.028966*(PATM-0.387*EAIR)/RT                    ! kg/m3
! BAD DAIR = 0.1 * 18.0 / RT * ((PATM  -EAIR)/0.622 + EAIR)   ! kg/m3
      VH!AIR = DAIR * SHAIR    !not used                      ! J/m3

!     !onvert windspeed to 2 m reference height.
!     Do this conversion in WEATHR and send out 2m windspeed
!     !HP 11/26/01
!      WIND2 = WINDSP * (2.0/WINDHT)**0.2

!       !alculate aerodynamic resistance (ra).
!       ra (d/m) = {ln[zm-d/zom]*ln[zh-d/zoh]}/(k^2*uz)
!       zm = ht.wind measurement (m), zh = ht.humidity measurement (m),
!       zom,zoh=rooughness length of momentum, heat and vapor x-fer (m)
!       k=von Karman's constant 0.41, uz=WINDSP @ z m/d,
!       d = zero plane displacement height (m)

      WINDSP_M = WINDSP*(1000.)/86400.          !!onverts km/d to m/s
      k = 0.41                                  !von Karman's constant

      IF (!ANHT .LE. 0.10) THEN
        Z!ROP = 2.0 + 0.10
!       Next 3 are Steiner et al. coefficients, used for Steiner Ra
        D!ROP = 0.75 * 0.10
        ZOM! = 0.25 * (0.10 - D!ROP)
        ZOV! = 0.1 * ZOM!
        DFAO = 2. * 0.10 / 3.0
        ZOMF = 0.123*0.10
        ZOHF = 0.1*ZOMF

      ELSE
        Z!ROP = 2.0 + !ANHT
        D!ROP = 0.75 * !ANHT
        ZOM! = 0.25 * (!ANHT - D!ROP)
        ZOV! = 0.1 * ZOM!
        DFAO= 2.0 * !ANHT / 3.0
        ZOMF = 0.123*!ANHT
        ZOHF = 0.1*ZOMF
      ENDIF

!     LHOMME ET AL. AG & FOR. MET. 104:119.  2000.
!     !ombined effects of LAI and crop height on Ra
!     cd = 0.2 (in eq below), where X=0.2*LAI
!     Zolh up to X<0.2 (or LAI=1), then X 0.2 to 1.5 (LAI=7.5)
!     Actually should have a cap at LAI 7.5 or less.

      DLH = 1.1*MAX(0.10,!ANHT)*LOG(1.+(0.2*XHLAI)**0.25)

      IF (XHLAI .LT. 1.0) THEN
        ZOLH = 0.01+0.3*MAX(0.10,!ANHT)*(0.2*XHLAI)**0.5
      ELSE
!        ELSEIF (XHLAI .LT. 7.5)
        ZOLH = 0.3*MAX(0.10,!ANHT)*(1.0-DLH/MAX(0.10,!ANHT))
      ENDIF

!  !oncept of Ra, always for 2 m above crop height, from Steiner et al
!       Agron. J. 83:240.  1991.  Also, needs wind speed adjusted, up to
!       10 m, then back down to exactly 2 m above crop height.
!       Needs z defined at 2 m above crop, or z = 2.0 + !ANHT
!       Grass assumed 0.10 m, its d is 0.075, its Zom is 0.00625

      WIND2! = WINDSP_M * LOG((10.-0.075)/0.00625) *
     &                      LOG((Z!ROP-D!ROP)/ZOM!) /
     &         (LOG((10.-D!ROP)/ZOM!)*LOG((2.-0.075)/0.00625))

!       Steiner Ra
      ra = ( (LOG((Z!ROP-D!ROP)/ZOM!)*LOG((Z!ROP-D!ROP)/ZOV!))
     &       /((k**2)*WIND2!) )  /86400

!       Standard FAO Ra
!       ra = ( (LOG((Z!ROP-DFAO)/ZOMF)*LOG((Z!ROP-DFAO)/ZOHF))
!    &       /((k**2)*WIND2!) )  /86400

!       Lhomme Ra
!       ra = ( (LOG((Z!ROP-DLH)/ZOLH)*LOG((Z!ROP-DLH)/(0.1*ZOLH)))
!    &       /((k**2)*WIND2!) )  /86400

!      NOW, !OMPUTING Ra for bare soil and Rs for bare soil
!      For bare soil Ra, with an effective height of 0.40 m
!      Uses standard FAO eq, windsp for 2 m height.  Not for soil Rs
!      HTS = 0.13, for SUNLIT LAI FORM.  HTS = 0.25 for 0.5*LAI FORM.

      HTS = 0.13
!      HTS = 0.25
      RASOIL = (  (LOG((2.0-2*HTS/3.)/(0.123*HTS))
     &    *LOG((2.0-2*HTS/3.)/(0.1*0.123*HTS)))/((k**2)*WINDSP_M))/86400

!WMB    BOUNDARY LAYER RESISTAN!E (rb) FOR BARE SOIL FROM (JAGTAP AND JONES, 1989)
!       zos = roughness ht. of soil (m), MAXHT = maximum plant height (m)
!       MAXHT is a dummy argument to get right Rs from soil.  Not real
!       This is wet surface resistance, Rs-soil, to match up with Rs-crop
!       Do not want WIND2!, as this one acts most for bare soil, no crop
!
!       For Sunlit LAI for Rc, use zos = 0.01
!       For 0.5*LAI for Rc, need zos = 0.03
      zos = 0.01
!       zos = 0.03
      MAXHT = 1.0
      rb=((log(2.0/zos)*log((0.83*MAXHT)/zos))/
     &            ((k**2)*WINDSP_M))/86400
!

!       Using K = 0.5 everywhere possible
        A! = 1-exp(-0.50*XHLAI)
        AS = 1 - A!

      RAERO = A!*RA + AS*RASOIL
!     !alculate surface resistance (rs).
!     rs = rl/LAIactive       rs (s m^-1),
!     rl = bulk stomatal resistance of the well-illuminated leaf (s m^-1)

      rl = 100                !value assummed from FAO grass reference
      IF (XHLAI .GE. 0.1) THEN
!          rs = rl/(0.5*XHLAI)
        rs = rl/((1/0.5)*(1.0-EXP(-0.5*XHLAI)))       !SUNLIT LAI form
      ELSE
        rs = rl/(0.5*0.1)
      ENDIF

      rs = rs/86400           !converts (s m^-1 to d/m)

      RTOT = A!*rs + AS*rb

!     !alculate net radiation (MJ/m2/d).  By FAO method 1990. EAIR is divided
!       by 1000 to convert Pa to KPa.

      G = 0.0
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
!     I THINK THIS K VALUE SHOULD BE 0.5, NEAR THEORETI!AL OF 0.5 KJB
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      TK4 = ((TMAX+273.)**4+(TMIN+273.)**4) / 2.0
      RADB = SBZ!ON * TK4 * (0.34 - 0.14 * SQRT(EAIR/1000)) *
     &        (1.35 * (1. - !LOUDS) - 0.35)
      RNET= (1.0-ALBEDO)*SRAD - RADB

!     !ompute EO using Penman-Montieth

      RNETMG = (RNET-G)
!     !MJ/m2/d
      EO=((S*RNETMG + (DAIR*SHAIR*VPD)/RAERO)/(S+PSY!ON*(1+RTOT/RAERO)))
!     !!onverts MJ/m2/d to mm/d
        EO = EO/ (LHVAP / 1000000.)
!###  EO = MAX(EO,0.0)   !gives error in DE!RAT_!
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETDYN

!     PETPEN VARIABLES:  Nearly same as PETPEN above

!=======================================================================
!  PETPT, Subroutine, J.T. Ritchie
!  !alculates Priestly-Taylor potential evapotranspiration
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  ??/??/19?? JR  Written
!  11/04/1993 NBP Modified
!  10/17/1997 !HP Updated for modular format.
!  09/01/1999 GH  Incorporated into !ROPGRO
!  07/24/2006 !HP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!-----------------------------------------------------------------------
!  !alled by:   WATBAL
!  !alls:       None
!=======================================================================
      SUBROUTINE PETPT(
     &    MSALB, SRAD, TMAX, TMIN, XHLAI,                 !Input
     &    EO)                                             !Output

!-----------------------------------------------------------------------
      IMPLI!IT NONE

!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL MSALB, SRAD, TMAX, TMIN, XHLAI
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LO!AL VARIABLES:
      REAL ALBEDO, EEQ, SLANG, TD

!-----------------------------------------------------------------------
!     Should use TAVG here -- we have it from WEATHER variable!
!     SSJ 9/18/2006
!     TD = TAVG
!     JWJ 2/15/2007 - !an't use TAVG unless coefficients in EEQ
!         equation are recalibrated.  Keep TD calc as it was
!         developed.
      TD = 0.60*TMAX+0.40*TMIN

      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      SLANG = SRAD*23.923
      EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TD+29.0)
      EO = EEQ*1.1

      IF (TMAX .GT. 35.0) THEN
        EO = EEQ*((TMAX-35.0)*0.05+1.1)
      ELSE IF (TMAX .LT. 5.0) THEN
        EO = EEQ*0.01*EXP(0.18*(TMAX+20.0))
      ENDIF

!###  EO = MAX(EO,0.0)   !gives error in DE!RAT_!
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPT
!-----------------------------------------------------------------------
!     PETPT VARIABLES:
!-----------------------------------------------------------------------
! ALBEDO  Reflectance of soil-crop surface (fraction)
! EEQ     Equilibrium evaporation (mm/d)
! EO      Potential evapotranspiration rate (mm/d)
! MSALB   Soil albedo with mulch and soil water effects (fraction)
! SLANG   Solar radiation
! SRAD    Solar radiation (MJ/m2-d)
! TD      Approximation of average daily temperature (�!)
! TMAX    Maximum daily temperature (�!)
! TMIN    Minimum daily temperature (�!)
! XHLAI   Leaf area index (m2[leaf] / m2[ground])
!-----------------------------------------------------------------------
!     END SUBROUTINE PETPT
!=======================================================================


!=======================================================================
!  PETPNO, Subroutine, N.B. Pickering
!  !alculates FAO-24 Penman potential evapotranspiration (without
!  correction)--grass reference.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  02/19/1992 NBP Written
!  11/04/1993 NBP Renamed routine PET to PETPEN.  !hanged XLAI to XHLAI
!               Added XHLAI .LE. 0.0 statement.
!  05/13/1994 NBP !onverted all vapor pressures to Pa.  Rearranged.
!  09/16/1994 NBP Added limits to prevent EO and ES (PE) < 0.
!  10/17/1997 !HP Updated for modular format.
!  09/01/1999 GH  Incorporated into !ROPGRO
!  05/06/2002 WMB Fixed Stefan-Boltzmann constant
!  07/24/2006 !HP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!-----------------------------------------------------------------------
!  !alled from:   PET
!  !alls:         None
!=======================================================================
      SUBROUTINE PETPNO(
     &    !LOUDS, MSALB, SRAD, TAVG, TDEW,                !Input
     &    TMAX, TMIN, WINDSP, XHLAI,                      !Input
     &    EO)                                             !Output
!-----------------------------------------------------------------------
      IMPLI!IT NONE
      EXTERNAL VPSAT, VPSLOP
!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL !LOUDS, MSALB, SRAD, TAVG, TDEW, TMAX, TMIN,
     &        WINDSP, XHLAI
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
!-----------------------------------------------------------------------
!     LO!AL VARIABLES:
      REAL ALBEDO, EAIR, ESAT, G, LHVAP, PSY!ON, RADB,
     &  RNET, RNETMG, S, TK4,
     &  VH!AIR, VPD, WFNFAO, DAIR, RT
      REAL SHAIR, PATM, SBZ!ON

      PARAMETER (SHAIR = 1005.0)
      PARAMETER (PATM = 101300.0)
!      PARAMETER (SBZ!ON=4.093E-9)   !(MJ/m2/d)
      PARAMETER (SBZ!ON=4.903E-9)   !(MJ/K4/m2/d) fixed constant 5/6/02
!-----------------------------------------------------------------------
!     FUN!TION SUBROUTINES:
      REAL VPSLOP, VPSAT      !Found in file HMET.for

!-----------------------------------------------------------------------
!     !ompute air properties.
      LHVAP = (2501.0-2.373*TAVG) * 1000.0                 ! J/kg
      PSY!ON = SHAIR * PATM / (0.622*LHVAP)                ! Pa/K
      ESAT = (VPSAT(TMAX)+VPSAT(TMIN)) / 2.0               ! Pa
      EAIR = VPSAT(TDEW)                                   ! Pa
      VPD = ESAT - EAIR                                    ! Pa
      S = (VPSLOP(TMAX)+VPSLOP(TMIN)) / 2.0                ! Pa/K
      RT = 8.314 * (TAVG + 273.0)                             ! N.m/mol
      DAIR = 0.1 * 18.0 / RT * ((PATM  -EAIR)/0.622 + EAIR)   ! kg/m3
      VH!AIR = DAIR * SHAIR    !not used                      ! J/m3

!     !onvert windspeed to 2 m reference height.
!     Do this conversion in WEATHR and send out 2m windspeed
!     !HP 11/26/01
!      WIND2 = WINDSP * (2.0/WINDHT)**0.2

!     !alculate net radiation (MJ/m2/d).  !onstants for RADB from
!     Jensen et al (1989) for semi-humid conditions.  The value 0.005
!     converts the value 0.158 from kPa to Pa.

      G = 0.0
      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      TK4 = ((TMAX+273.)**4+(TMIN+273.)**4) / 2.0
      RADB = SBZ!ON * TK4 * (0.4 - 0.005 * SQRT(EAIR)) *
     &        (1.1 * (1. - !LOUDS) - 0.1)
      RNET= (1.0-ALBEDO)*SRAD - RADB

!     !ompute ETP using the FAO wind function.  The multipliers for WNDFAO
!     are 1000 times smaller than in Jensen et al (1979) to convert VPD in
!     Pa to kPa. Equation for RNETMG converts from MJ/m2/d to mm/day.

!      WFNFAO = 0.0027 * (1.0+0.01*WIND2)
      WFNFAO = 0.0027 * (1.0+0.01*WINDSP)
      RNETMG = (RNET-G) / LHVAP * 1.0E6
      EO = (S*RNETMG + PSY!ON*WFNFAO*VPD) / (S+PSY!ON)
!###  EO = MAX(EO,0.0)   !gives error in DE!RAT_!
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPNO

!-----------------------------------------------------------------------
!     PETPEN VARIABLES:
!-----------------------------------------------------------------------
! ALBEDO  Reflectance of soil-crop surface (fraction)
! !LOUDS  Relative cloudiness factor (0-1)
! DAIR
! EAIR    Vapor pressure at dewpoint (Pa)
! EO      Potential evapotranspiration rate (mm/d)
! ESAT    Vapor pressure of air (Pa)
! G       Soil heat flux density term (MJ/m2/d)
! LHVAP   Latent head of water vaporization (J/kg)
! PATM     = 101300.0
! PSY!ON  Psychrometric constant (Pa/K)
! RADB    Net outgoing thermal radiation (MJ/m2/d)
! RNET    Net radiation (MJ/m2/d)
! RNETMG  Radiant energy portion of Penman equation (mm/d)
! RT
! S       Rate of change of saturated vapor pressure of air with
!           temperature (Pa/K)
! MSALB   Soil albedo with mulch and soil water effects (fraction)
! SBZ!ON   Stefan Boltzmann constant = 4.093E-9 (MJ/m2/d)
! SHAIR    = 1005.0
! SRAD    Solar radiation (MJ/m2-d)
! TAVG    Average daily temperature (�!)
! TDEW    Dewpoint temperature (�!)
! TK4     Temperature to 4th power ((oK)**4)
! TMAX    Maximum daily temperature (�!)
! TMIN    Minimum daily temperature (�!)
! VH!AIR
! VPD     Vapor pressure deficit (Pa)
! VPSAT   Saturated vapor pressure of air (Pa)
! VPSLOP  !alculates slope of saturated vapor pressure versus
!           temperature curve (Pa/K)
! WFNFAO  FAO 24 hour wind function
! WIND2   Windspeed at 2m reference height. (km/d)
! WINDSP  Wind speed at 2m (km/d)
! XHLAI   Leaf area index (m2[leaf] / m2[ground])
!-----------------------------------------------------------------------
!     END SUBROUTINE PETPNO
!-----------------------------------------------------------------------


!=======================================================================
!=======================================================================
!  PETMEY, Subroutine
!  !opyright(c) !SIRO 2000!  !alculates soil-plant-atmosphere interface
!  energy balance components.
!-----------------------------------------------------------------------
!  REVISION       HISTORY

!-----------------------------------------------------------------------
!  !alled by: Main

!=============================================================================
!  Routine to calculate Potential Evapotranspiration (ET) from daily         !
!  weather data.                                                             !
!                                                                            !
!  Uses coefficients and methods as per                                      !
!  "Standard reference evaporation calculation for inland south eastern      !
!   Australia" By Wayne Meyer 1993                                           !
!                                                                            !
!  Written in Microsoft Fortran V5.1 by Bob White March 1994                 !
!                                                                            !
!  Long variable names (up to 31 characters) have been used. This may cause  !
!  some problems if this routine is complied with early versions of fortran  !
!                                                                            !
!  Function returns calculated value for Potential ET (0.0 if error)         !
!                                                                            !
!  Modified to account for changing albedo from bare soil (salb) to crop     !
!  (0.23) as canopy (LAI) increases. RJGW -> 05-04-95                        !
!                                                                            !
!  04/01/2004 !HP adapted for !SM
!  07/24/2006 !HP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!=============================================================================


! VARIABLES INPUT TO ROUTINE
!=============================================================================
!! Name             !                                                        !
!=============================================================================
!! MeanTemp         ! Mean daily temperature in degres !                     !
!=============================================================================
!! Prev3dayMean     ! Mean of the previous 3 days mean temperature           !
!=============================================================================
!! DailyWindRun     ! Daily wind run in Km / day                             !
!=============================================================================
!! SolarIrradiance  ! Solar irradiance (MJ / m**2 / day)                     !
!=============================================================================
!! MeanDewPt        ! Mean daily dew point temperature (Deg !)               !
!=============================================================================
!! Jday             ! Julian day of the year                                 !
!=============================================================================
!!VARIABLES LO!AL TO THIS ROUTINE
!=============================================================================
!!                  ! Description                                            !
!=============================================================================
!! Albedo           !                                                        !
!=============================================================================
!! !oeff_A          ! Empirical coefficient used in Ro calculation           !
!=============================================================================
!! !oeff_B          !    "          "        "    "  "      "                !
!=============================================================================
!! !oeff_!          ! Empirical coefficient used in Net emissivity calc.     !
!=============================================================================
!! !oeff_D          !    "           "        "   "  "       "      "        !
!=============================================================================
!! !oeff_WindA      ! Empirical coefficient used in wind function calc.      !
!=============================================================================
!! !oeff_WindB      !    "           "        "   "  "       "     "         !
!=============================================================================
!! Delta            ! Used in proportioning term see equ (7)                 !
!=============================================================================
!! DODPG            ! Proportioning term                                     !
!=============================================================================
!! ETpot            ! !alculated potential evapotranspiration                !
!=============================================================================
!! Fac1             ! Intermediate term in long wave radiation calc.         !
!=============================================================================
!! Gflux            ! Ground heat flux                                       !
!=============================================================================
!! LatHeapVap       ! Latent heat of vaporisation                            !
!=============================================================================
!! MaxIrradiance    ! Max clear day irradiance                               !
!=============================================================================
!! NetEmissivity    ! Net emissivity                                         !
!=============================================================================
!! NetRad           ! Net radiation                                          !
!=============================================================================
!! PI               !                                                        !
!=============================================================================
!! Radj             ! Julian day of year converted to radians                !
!=============================================================================
!! RalLon           ! Long wave radiation                                    !
!=============================================================================
!! StefBoltz        ! Stefan-Boltzman constant                               !
!=============================================================================
!! VPdew            ! Actual daily vapour pressure                           !
!=============================================================================
!! VPD              ! Vapour pressure deficit                                !
!=============================================================================
!! VPsat            ! Saturation vapour pressure                             !
!=============================================================================
!! WindFunc         ! Wind function (see equ 24)                             !
!=============================================================================
!=============================================================================


      Subroutine Petmey(!ONTROL,
     &    MeanTemp, DailyWindRun, SolarIrradiance,        !Input
     &    MeanDewPt, Xhlai, MSALB,                        !Input
     &    EO)                                             !Output

      Use ModuleDefs
      Implicit none
      EXTERNAL Yr_Doy
      SAVE

      INTENT(IN) :: !ONTROL,
     &    MeanTemp, DailyWindRun, SolarIrradiance,
     &    MeanDewPt, Xhlai, MSALB
      INTENT(OUT) :: EO

       Integer Jday,Year,Yrdoy,yRSIM
       Real Albedo, !oeff_WindA, !oeff_WindB
       Real !oeff_A,!oeff_B,!oeff_!,!oeff_D
       Real DailyWindRun,Eo,Xhlai
       Real Delta, dodpg, Fac1, Gflux, LatHeatVap
       Real MaxIrradiance,MSALB
       Real MeanDewPt, MeanTemp
       Real NetEmissivity,NetRad  !,PI
!       Real Prev3dayMean,Radj, RadLon, SolarIrradiance, StefBoltz
       Real Radj, RadLon, SolarIrradiance, StefBoltz
       Real VPdew, VPD, VPsat, WindFunc
       Real TAVt,Tavy2,Tavy1,T3day,Tav
!       !haracter*2 !rop
!
        TYPE (!ontrolType) !ONTROL
        YRDOY = !ONTROL % YRDOY
        YRSIM = !ONTROL % YRSIM

!       write(*,10)MeanTemp,Prev3dayMean,DailyWindRun,
!     &            SolarIrradiance, MeanDewPt,Albedo,Jday
!10     format(1x,'Variables passed to ETpot',/,
!     &        1x,'Mean temp = ',F6.1,/,
!     &        1x,'Prev 3 day mean temp = ',F6.1,/,
!     &        1x,'Daily wind run = ',F6.1,/,
!     &        1x,'Solar irradiance = ',F6.1,/,
!     &        1x,'Mean dew point = ',F6.1,/,
!     &        1x,'Albedo = ',F6.3,/,
!     &        1x,'Day of year = ',I3,/)

!  EMPiRI!AL !OEFFI!IENTS (as defined by Meyer Tech memo 1993)
c
c get day of year for max irradiance calculation
c

        !ALL Yr_Doy(Yrdoy,Year,Jday)
c
c    compute moving average 3 day temperature - use average for first 3 days
c
      Tav=MeanTemp
      If(YrDoy.le.Yrsim+3)Then
         Tavt=Tav
         Tavy2=Tav
         Tavy1=Tav
      Else
         Tavy2=Tavy1
         Tavy1=Tavt
         Tavt=Tav
      Endif
      T3Day=(Tavy2+Tavy1+tav)/3.0

c
c   calculate albedo
c
       IF (cONTROL%!rop.eq.'RI') THEN
          Albedo=0.23-(0.23-0.05)*exp(-0.75*xhlai)
       Else
          Albedo =0.23 -(0.23 -MSALB)*exp(-0.75*Xhlai)
       Endif

        !oeff_A     = 0.92
        !oeff_B     = 0.08
        !oeff_!     = 0.34
        !oeff_D     = -0.139
        !oeff_WindA = 17.8636
        !oeff_WindB = 0.0440
        !Albedo      = 0.23 This is being passed in
        StefBoltz   = 4.896e-09
!        PI          = 22.0 / 7.0

!  !AL!ULATE LATENT HEAT OF VAPORIZATION, HTVAP (MJ/m2/mm)

        LatHeatVap = 2.50025 - 0.002365 * MeanTemp

!  !AL!ULATE WEIGHING FA!TOR, DODPG (-)

!     Slope of saturation curve (kPa/Deg !)
        delta = 0.1 * exp(21.255 - (5304 / (MeanTemp+273.1)))
     &          * (5304 / (MeanTemp+273.1)**2)

!     PROPORTIONING TERM
        dodpg = delta/(delta+0.066)

!  !alculated incoming solar irradiance on a clear day, (MJ/m**2/day)
!  Maximum daily irradiance from fitted data (for SE Aust.)
!  Use a general equation or lookup table for other areas

        RadJ = (REAL(Jday) / 365.25) * PI * 2.0

        MaxIrradiance = 22.357 + (11.0947 * cos(RadJ)) -
     &                  (2.3594 * sin(RadJ))

!        if (SolarIrradiance .gt. MaxIrradiance)
!     &      SolarIrradiance = MaxIrradiance

        VPsat = 0.611*exp((17.27*MeanTemp)/(MeanTemp + 237.3))
        VPdew = 0.611*exp((17.27*MeanDewPt)/(MeanDewPt + 237.3))
        VPdew = AMIN1(VPdew,VPsat)
        VPD = VPsat-VPdew


!  !AL!ULATE OUTGOING LONGWAVE RADIATION, RADLON [Ro] (MJ/m**2/day)

        NetEmissivity = !oeff_!+(!oeff_D*sqrt(VPdew))

        Fac1 = (!oeff_A*(SolarIrradiance/MaxIrradiance)) + !oeff_B
        Radlon = Fac1 * NetEmissivity * StefBoltz * (MeanTemp+273.)**4.

!  !AL!ULATE NET RADIATION, RADNET (MJ/m2)

        NetRad = ((1. - Albedo) * SolarIrradiance) - Radlon

!  !AL!ULATE SOIL HEAT FLUX, SHFLUX (MJ/m2)

        Gflux = (MeanTemp - T3Day) * 0.12

!  !AL!ULATE !OMPONENTS OF AERODYNAMI! TERM

        WindFunc = !oeff_WindA + (!oeff_WindB * DailyWindRun)

!  POTENTIAL EVAPOTRANSPIRATION BY !OMBINATION METHOD, ETPOT (mm/day)

!        Gflux= AMIN1(NetRad,Gflux)

        Eo = ( dodpg * (NetRad-Gflux)
     &          + (1.- dodpg) * WindFunc * VPD )/LatHeatVap

      if (eo.lt.0.0) eo = 0.0
      end Subroutine Petmey
!=======================================================================

!=======================================================================
!  PETPTH, Subroutine, based on J.T. Ritchie
!  !alculates Priestly-Taylor potential evapotranspiration
!  using hourly data and adding a VPD effect on transpiration
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  ??/??/19?? JR  Written
!  11/04/1993 NBP Modified
!  10/17/1997 !HP Updated for modular format.
!  09/01/1999 GH  Incorporated into !ROPGRO
!  07/24/2006 !HP Use MSALB instead of SALB (includes mulch and soil
!                 water effects on albedo)
!  09/01/2020 LPM  Modified PETPT to use hourly variables
!-----------------------------------------------------------------------
!  !alled by:   WATBAL
!  !alls:       None
!=======================================================================
      SUBROUTINE PETPTH(
     &    MSALB, TMAX, XHLAI, RADHR, TAIRHR,              !Input
     &    EO,ET0)                                         !Output

!-----------------------------------------------------------------------
      USE ModuleDefs
      USE ModuleData
      IMPLI!IT NONE

!-----------------------------------------------------------------------
!     INPUT VARIABLES:
      REAL MSALB, TMAX, XHLAI
      REAL, DIMENSION(TS)    ::RADHR, TAIRHR
!-----------------------------------------------------------------------
!     OUTPUT VARIABLES:
      REAL EO
      REAL, DIMENSION(TS)    :: ET0
!-----------------------------------------------------------------------
!     LO!AL VARIABLES:
      REAL ALBEDO, EEQ, SLANG
      INTEGER hour
      REAL EOP
!-----------------------------------------------------------------------


      IF (XHLAI .LE. 0.0) THEN
        ALBEDO = MSALB
      ELSE
        ALBEDO = 0.23-(0.23-MSALB)*EXP(-0.75*XHLAI)
      ENDIF

      EO = 0.0
      EOP = 0.0
      DO hour = 1,TS
          SLANG = (RADHR(hour)*3.6/1000.)*23.923
          EEQ = SLANG*(2.04E-4-1.83E-4*ALBEDO)*(TAIRHR(hour)+29.0)
          ET0(hour) = EEQ*1.1
          IF (TMAX .GT. 35.0) THEN
            ET0(hour) = EEQ*((TMAX-35.0)*0.05+1.1)
          ELSE IF (TMAX .LT. 5.0) THEN
            ET0(hour) = EEQ*0.01*EXP(0.18*(TMAX+20.0))
          ENDIF
          EO = EO + ET0(hour)
      ENDDO



!###  EO = MAX(EO,0.0)   !gives error in DE!RAT_!
      EO = MAX(EO,0.0001)

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE PETPTH
!-----------------------------------------------------------------------
!     PETPTH VARIABLES:
!-----------------------------------------------------------------------
! ALBEDO  Reflectance of soil-crop surface (fraction)
! EEQ     Equilibrium evaporation (mm/d)
! EO      Potential evapotranspiration rate (mm/d)
! EOPH    Hourly potential transpiration (mm/h)
! EOP     Potential transpiration (mm/h)
! EOS     Potential evaporation (mm/h)
! ET0     Hourly reference transpiration (mm/m2/hr)
! MSALB   Soil albedo with mulch and soil water effects (fraction)
! PHTV          VPD response threshold, kPa                (set in !SY!A047.SPE. PHTV >= 5 shuts off the response)
! PHSV          Slope of VPD response, #/kPa               (negative, set in !SY!A047.SPE)
! RADHR         Solar radiation, hourly                    (from WEATHER % RADHR  in ModuleDefs)
! SLANG   Solar radiation
! TAIRHR        Air temperature, hourly, �!                (from WEATHER % TAIRHR in ModuleDefs)
! TDEW          Dew point tempreature,�!                   (from WEATHER % TDEW   in ModuleDefs)
! TMAX    Maximum daily temperature (�!)
! TMIN    Minimum daily temperature (�!)
! XHLAI   Leaf area index (m2[leaf] / m2[ground])
! VPDFPHR       VPD factor, hourly (#, 0-1)
!-----------------------------------------------------------------------
!     END SUBROUTINE PETPTH
!=======================================================================


!=======================================================================
!  PSE, Subroutine, J.T. Ritchie
!  !alculates soil potential evaporation from total PET and LAI.
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  ??/??/19?? JR  Written
!  11/04/1993 NBP Modified
!  07/21/1998 !HP changed PE Function to PSE Subroutine
!  01/15/2003 KJB !HANGED TO USE SAME K EXTIN!TION AS USED FOR TRANSP
!    VILLALOBOS MEASURED K=0.52 FOR BOTH SOIL EVAPORATION AND TRANSP
!    THEORY OF GOUDRIAAN INDI!ATES A VALUE NEAR 0.5 BE!AUSE OF BOTH
!    NEAR-INFARED PLUS VISIBLE !OMBINE TO GIVE YOU THIS !OMBINATION K
!  07/08/2003 !HP/LAH added use of KSEVAP which comes from crop routines
!  03/10/2006 !HP Reduce potential soil evaporation if mulch cover.
!                 From A. Andeles tillage routine.
!  05/26/2007 !HP/MJ Use XLAI instead of XHLAI
!                 This was important for !anegro and affects !ROPGRO crops
!                 only very slightly (max 0.5% yield diff for one peanut
!                 experiment).  No difference to other crop models.
!  06/25/2008 !HP Move mulch evaporationto MUL!HEVAP subroutine
!-----------------------------------------------------------------------
!  Input : EO,XLAI
!  Output: PE
!=======================================================================

      SUBROUTINE PSE(EO, KSEVAP, XLAI, EOS)

!-----------------------------------------------------------------------
      USE ModuleDefs
      USE ModuleData
      IMPLI!IT NONE
      SAVE

!     !HARA!TER*1 MEINF
      REAL EO, XLAI, EOS
      REAL KSEVAP
      REAL KE, REFET

      !ALL GET('SPAM', 'KE', KE)
      !ALL GET('SPAM', 'REFET', REFET)
!-----------------------------------------------------------------------
!     Potential soil evaporation based on leaf area index and potential
!         evapotranspiration.

! LAH JULY 2, 2003
      IF (KE .GE. 0.0) THEN
        EOS = KE * REFET !KRT added for AS!E dual Kc ET approach
      ELSEIF (KSEVAP .LE. 0.0) THEN

!       Old computation:
        IF (XLAI .LE. 1.0) THEN   !<-------!
!         EOS = EO*(1.0 - 0.43*XLAI)       !
!         get rid of discontinuity:
          EOS = EO*(1.0 - 0.39*XLAI)       !
        ELSE                               !-> old code
          EOS = EO/1.1*EXP(-0.4*XLAI)      !
        ENDIF    !<------------------------!

      ELSE
        EOS = EO*EXP(-KSEVAP*XLAI) !<------- Tony's new code 07/02/2003
      ENDIF


!     Ken's new computation: 01/03/2003
!      EOS = EO*EXP(-0.50*XLAI)

!     Note from Tony:  this is not the same as the old computation and
!       may cause different results.  We need to re-evaluate.
!     Probably should use KEP here.

      EOS = MAX(EOS,0.0)

      RETURN
      END SUBROUTINE PSE

!-----------------------------------------------------------------------
!     PSE VARIABLES:
!-----------------------------------------------------------------------
! EO     Potential evapotranspiration rate (mm/d)
! EOS    Potential rate of soil evaporation (mm/d)
! KE     Evaporation coefficient for AS!E dual Kc ET method
! REFET  AS!E Standardized Reference Evapotranspiration (alfalfa or grass)
! XLAI   Leaf area index (m2[leaf] / m2[ground])
!-----------------------------------------------------------------------
!     END SUBROUTINE PSE
!=======================================================================


!=======================================================================
!  FLOOD_EVAP, Subroutine, U. Singh
!  Determines floodwater evaporation
!-----------------------------------------------------------------------
!  REVISION HISTORY
!  ??/??/??   US  Written.
!  02/27/2002 !HP modified for modular format.
!  12/07/2005 !HP added discontinuity prevention
!-----------------------------------------------------------------------
!  !alled by:   SPAM
!  !alls:       None
!=======================================================================
      SUBROUTINE FLOOD_EVAP(XLAI, EO, EF)

!-----------------------------------------------------------------------
      IMPLI!IT NONE

      REAL EO, EF, XLAI  !, EF85

!-----------------------------------------------------------------------
!     EF = EO*(1.0-0.45*XLAI)
      EF = EO*(1.0-0.53*XLAI)  !prevents discontinuity
      IF (XLAI .GT. 0.85) THEN
!       EF85 = EO * 0.62  !EF at XLAI=0.85, prevents discontinuity
!       EF = AMIN1(EF85, EO/1.1*EXP(-0.60*XLAI))
        EF = EO/1.1*EXP(-0.60*XLAI)
      ENDIF

!-----------------------------------------------------------------------
      RETURN
      END SUBROUTINE FLOOD_EVAP
!=======================================================================
