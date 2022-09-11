!===============================================================================
!     Fortranification of the CHIME SIR disease model for COVID outbreak
!     Euler method simulation
!===============================================================================

!===============================================================================
! GET_BETA, subroutine, by P. Hein
! Calculates a rate of exposure given an intrinsic growth rate for COVID-19
!-------------------------------------------------------------------------------
!     Input Variables:
!     intrinsic_growth_rate   Rate of spread of COVID-19 cases
!     gamma         The expected recovery rate from COVID-19 for infected individuals
!     s_c           Current amount of individuals that are susceptible
!     contact_rate  The relative contact rate amongst individuals in the population
!
!     State Variables:
!     inv_contact_rate      The inverse rate of contact between individuals in the population
!     updated_growth_rate   The intrinsic growth rate adjusted for the recovery rate from infection
!
!     Output Variables:
!     beta              The rate of exposure of individuals to persons infected with COVID-19
!
!-------------------------------------------------------------------------------
!  Called by:   main
!  Calls:       None
!==============================================================================
      subroutine get_beta(intrinsic_growth_rate, gamma,
     &                    s_c, relative_contact_rate,
     &                    beta)
        real intrinsic_growth_rate, s_c, relative_contact_rate
        real inv_contact_rate, updated_growth_rate, gamma, beta

        inv_contact_rate = 1.0 - relative_contact_rate
        updated_growth_rate = intrinsic_growth_rate + gamma
        beta = updated_growth_rate / s_c * inv_contact_rate
      end subroutine get_beta

!===============================================================================
! GET_GROWTH_RATE, subroutine, by P. Hein
! Calculate the expected growth rate of COVID-19 infections given a doubling time
!-------------------------------------------------------------------------------
!     Input/Output Variables:
!     growth_rate       Rate of spread of COVID-19 cases
!
!     Input Variables:
!     doubling_time     The time required for the amount of COVID-19 cases to double
!
!-------------------------------------------------------------------------------
!  Called by:   main
!  Calls:       None
!==============================================================================
      subroutine get_growth_rate(doubling_time, growth_rate)
        real doubling_time, growth_rate

        if (doubling_time .eq. 0.0) then
            growth_rate = 0.0
        else
          growth_rate = 2.0 ** (1.0 / doubling_time) - 1.0
        endif
      end subroutine get_growth_rate

!===============================================================================
!  SIR, Subroutine, P. Hein
!  Updates all disease states given the current state values
!-------------------------------------------------------------------------------
!     Input Variables:
!     s_c         Current amount of individuals that are susceptible
!     i_c         Current amount of individuals that are infectious
!     r_c         Current amount of individuals that are recovered
!     beta        The rate of exposure of indidivuals to persons infected with COVID-19
!     gamma       Rate of recovery for infected individuals
!     n           Total population size
!
!     State Variables:
!     s_n         Update to the amount of individuals that are susceptible
!     i_n         Update to the amount of individuals that are infectious
!     r_n         Update to the amount of individuals that are recovered
!     scale       A scaling factor to compute updated disease variables
!
!-------------------------------------------------------------------------------
!  Called by:   sim_sir
!  Calls:       None
!==============================================================================
      subroutine sir(s_c, i_c, r_c, beta, gamma, n)
        real s_c, i_c, r_c, n, s_n, i_n, r_n, beta, gamma, scale
        s_n = (-beta * s_c * i_c) + s_c
        i_n = (beta * s_c * i_c - gamma * i_c) + i_c
        r_n = gamma * i_c + r_c

        scale = n / (s_n + i_n + r_n)

        s_c = s_n * scale
        i_c = i_n * scale
        r_c = r_n * scale
      end subroutine sir

!===============================================================================
! SIM_SIR, subroutine, by P. Hein
! Simulates a COVID-19 outbreak where a policy intervention is attempted in
! order to lower the relative contact rate amongst individuals.
!-------------------------------------------------------------------------------
!     Input/Output Variables:
!     s_c         Current amount of individuals that are susceptible
!     i_c         Current amount of individuals that are infectious
!     r_c         Current amount of individuals that are recovered
!     T           An array used to record the current day during the simulation
!     S           An array used to record the susceptible population changes during the simulation
!     EI          An array used to record the total infected population changes during the simulation
!     I           An array used to record the currently infected population changes during the simulation
!     R           An array used to record the recovered population changes during the simulation
!
!     Input Variables:
!     gamma       The expected recovery rate from COVID-19 for infected individuals
!     i_day       Start day of COVID-19 infections
!     N_p         Number of policies to use for the simulation
!     N_t         Total length of simulation
!     betas       An array of beta values with one entry per beta
!     days        An array of time periods with one entry per policy
!
!     State Variables:
!     n           Total population size
!     beta        The rate of exposure of indidivuals to persons infected with COVID-19
!     N_d         The amount of days for for the current policy to be simulated
!     d           Tracks the current day in the simulation
!     p_idx       Index to use for accessing policy variables
!     d_idx       Index to be used for days in the current policy
!     idx         Index to be used to access arrays for storing simulated outputs
!     total_inf   The total population that is ever infected at a timestep during the simulation
!
!-------------------------------------------------------------------------------
!  Called by:   main
!  Calls:       sir
!==============================================================================
      subroutine sim_sir(s_c, i_c, r_c, gamma, i_day,
     &                   N_p, N_t, betas, days, T, S, EI, I, R)
        real s_c, i_c, r_c, n, gamma, beta
        integer d, i_day, idx, d_idx, p_idx, t_idx, total_inf
        integer N_d, N_p, N_t, T(N_t), days(N_p)
        real S(N_t), EI(N_t), I(N_t), R(N_t) betas(N_p)

        n = s_c + i_c + r_c
        d = i_day

        idx = 1
        do p_idx = 1, N_p
          beta = betas(p_idx)
          N_d = days(p_idx)
          do d_idx = 1, N_d
            T(idx) = d
            S(idx) = s_c
            I(idx) = i_c
            R(idx) = r_c
            idx = idx + 1
            call sir(s_c, i_c, r_c, beta, gamma, n)
            d = d + 1
          enddo
        enddo

        T(idx) = d
        S(idx) = s_c
        I(idx) = i_c
        R(idx) = r_c

        do t_idx = 1, N_t
          total_inf = I(t_idx) + R(t_idx)
          EI(t_idx) = total_inf
        enddo
      end subroutine sim_sir

      program main
        implicit none
        real s_c, i_c, r_c, beta, doubling_time, growth_rate
        integer p_idx, days_for_policy
        integer, parameter :: i_day = 17, n_days = 20
        integer, parameter :: N_p = 3, N_t = 121, infectious_days = 14
        real, parameter :: relative_contact_rate = 0.05
        real, parameter :: gamma = 1.0 / infectious_days
        real, dimension(1:N_p) :: policy_betas
        integer, dimension(1:N_p) :: policy_days
        integer, dimension(1:N_t) :: T
        real, dimension(1:N_t) :: S, EI, I, R

        s_c = 1000
        i_c = 0
        r_c = 0

        do p_idx = 1, N_p
          doubling_time = (p_idx - 1.0) * 5.0
          call get_growth_rate(doubling_time, growth_rate)
          call get_beta(growth_rate, gamma, s_c,
     &                  relative_contact_rate, beta)
          policy_betas(p_idx) = beta
          days_for_policy = n_days * p_idx
          policy_days(p_idx) = days_for_policy
        end do

        call sim_sir(s_c, i_c, r_c, gamma, i_day,
     &               N_p, N_t, policy_betas, policy_days,
     &               T, S, EI, I, R)
        print*, s_c, i_c, r_c
        print*, EI
      end program main
