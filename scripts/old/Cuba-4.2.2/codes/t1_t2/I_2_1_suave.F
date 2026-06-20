! -*- f90 -*-
!=============================================================================
! I_D.F
!
! Integral:
!   I = INT_0^{2pi} dphi  INT_0^inf  k dk  [T1 - T2]
!
!   T1 = alpha_D(qkp2) * alpha_D(qkm2) * Gp(q2)^2
!   T2 = alpha_D(qkp2) * alpha_D(qkm2) * Gp(f) * [2*Gp(q2) - Gp(f)]
!
!   alpha_D(s) = 1 / { b0 * [s + m2log(s)] * ln([s+4*m2log(s)]/L^2) }
!   m2log(s)   = mg^2 * { ln[(s+rho*mg^2)/L^2] / ln[rho*mg^2/L^2] }^(-1-g1)
!   Gp(s)      = exp(-a1*s - a2*s^2)
!   qkp2       = q2/4 + sqrt(q2)*k*cos(phi) + k^2
!   qkm2       = q2/4 - sqrt(q2)*k*cos(phi) + k^2
!   f          = q2 + 9*|k^2 - q2/4|
!
! Variable substitution (maps k in [0,inf) to t in [0,1)):
!   k = t/(1-t),  dk/dt = 1/(1-t)^2
!   phi = 2*pi*r, r in [0,1]
!
! COMPILE:
!   gfortran -ffree-form -O2 -o I_D I_D.F -lcuba -lm
!=============================================================================


!=============================================================================
! MODULE params_mod
! All physical parameters are here. Edit only this block.
!=============================================================================
module params_mod
  implicit none

  ! ---- PHYSICAL PARAMETERS (edit these) ------------------------------------
  real(8), parameter :: Q2     = 4.0d0       ! external momentum squared [GeV^2]
  real(8), parameter :: MG     = 0.380d0       ! gluon mass parameter mg [GeV]
  real(8), parameter :: LAMBDA = 0.284d0      ! QCD scale Lambda [GeV]
  real(8), parameter :: RHO    = 4.0d0       ! IR regulator rho in m2log
  real(8), parameter :: GAMMA1 = 0.084d0       ! exponent gamma1 in m2log
  real(8), parameter :: B0     = 0.71619724391D0   ! beta-function coeff b0
  real(8), parameter :: A1     = 1.491d0       ! Gp first coeff  [GeV^-2]
  real(8), parameter :: A2     = 2.77d0      ! Gp second coeff [GeV^-4]
  real(8), parameter :: SQRT_S     = 7000.0d0      ! Gp second coeff [GeV^-4]
  ! --------------------------------------------------------------------------

  ! Derived (do not edit)
  real(8), parameter :: Q_VAL  = sqrt(Q2)    ! |q| = sqrt(q2)
  
  ! Precompute constants for alpha_D
  real(8), parameter :: MG2     = MG * MG
  real(8), parameter :: LAMBDA2 = LAMBDA * LAMBDA
  real(8), parameter :: RHO_MG2 = RHO * MG2
  real(8), parameter :: LOG_DEN = dlog(RHO_MG2 / LAMBDA2)

end module params_mod


!=============================================================================
! PROGRAM integral_ID
!=============================================================================
program integral_ID
  use params_mod
  implicit none

  ! Cuba CUHRE settings
  integer,   parameter :: NDIM      = 2
  integer,   parameter :: NCOMP     = 1
  integer,   parameter :: NVEC      = 1
  real(8),   parameter :: EPSREL    = 1.0d-2
  real(8),   parameter :: EPSABS    = 1.0d-16
  integer,   parameter :: VERBOSE   = 0
  integer,   parameter :: LAST      = 4
  integer,   parameter :: SEED      = 0
  integer,   parameter :: MINEVAL   = 100
  integer,   parameter :: MAXEVAL   = 1000
  integer,   parameter :: NNEW      = 500
  integer,   parameter :: NMIN      = 2
  real(8),   parameter :: FLATNESS  = 75.0d0
  character(len=1), parameter :: STATEFILE = ' '
  integer :: SPIN = -1

  ! Cuba output
  integer :: nregions, neval, fail
  real(8) :: integral(NCOMP), error(NCOMP), prob(NCOMP)

  external :: integrand_ID

  ! Call Cuba CUHRE
  call suave(NDIM, NCOMP, integrand_ID, 0, NVEC,          &
             EPSREL, EPSABS, VERBOSE + LAST, SEED,         &
             MINEVAL, MAXEVAL, NNEW, NMIN, FLATNESS,       &
             STATEFILE, SPIN,                              &
             nregions, neval, fail, integral, error, prob)

  ! Print results
  write(*,*)
  write(*,'(a)') '  +----------------------------------------------------------+'
  write(*,'(a)') '  |           Cuba CUHRE  --  Result  I_D                    |'
  write(*,'(a)') '  +----------------------------------------------------------+'
  write(*,'(a,f10.5,a)') '  |   Q2      = ', Q2,     '  [GeV^2]                   |'
  write(*,'(a,f10.5,a)') '  |   mg      = ', MG,     '  [GeV]                     |'
  write(*,'(a,f10.5,a)') '  |   Lambda  = ', LAMBDA, '  [GeV]                     |'
  write(*,'(a,f10.5,a)') '  |   rho     = ', RHO,    '                            |'
  write(*,'(a,f10.5,a)') '  |   gamma1  = ', GAMMA1, '                            |'
  write(*,'(a,f10.5,a)') '  |   b0      = ', B0,     '                            |'
  write(*,'(a,f10.5,a)') '  |   a1      = ', A1,     '  [GeV^-2]                  |'
  write(*,'(a,f10.5,a)') '  |   a2      = ', A2,     '  [GeV^-4]                  |'
  write(*,'(a)') '  +----------------------------------------------------------+'
  write(*,'(a,f20.12,a)') '  |   Result        ', integral(1), '          |'
  write(*,'(a,es14.4,a)') '  |   Est. error    ', error(1),    '                     |'
  write(*,'(a,f10.6,a)')  '  |   chi2 prob     ', prob(1),     '                     |'
  write(*,'(a,i12,a)')    '  |   Evaluations   ', neval,        '                     |'
  write(*,'(a,i12,a)')    '  |   Subregions    ', nregions,     '                     |'
  if (fail == 0) then
    write(*,'(a)') '  |   Status        CONVERGED                                |'
  else
    write(*,'(a)') '  |   Status        WARNING: max evals reached               |'
  end if
  write(*,'(a)') '  +----------------------------------------------------------+'
  write(*,*)

end program integral_ID


!=============================================================================
! SUBROUTINE integrand_ID
! Called by Cuba at each integration point (xx(1), xx(2)) in [0,1]^2.
!=============================================================================
subroutine integrand_ID(ndim, xx, ncomp, ff, userdata)
  use params_mod
  implicit none

  integer, intent(in)  :: ndim
  real(8), intent(in)  :: xx(ndim)
  integer, intent(in)  :: ncomp
  real(8), intent(out) :: ff(ncomp)
  integer, intent(in)  :: userdata

  real(8), parameter :: TWOPI = 6.28318530717958647692d0

  real(8) :: t, r, k, phi, jac
  real(8) :: qkp2, qkm2, f_arg
  real(8) :: aD_p, aD_m
  real(8) :: Gp_q2, Gp_f
  real(8) :: T1val, T2val
  
  ! Local variables for alpha_D calculation
  real(8) :: log_num, expo, m2s, arg_log_aD, log_aD, denom_aD

  t = xx(1)
  r = xx(2)

  k   = SQRT_S * t
  phi = TWOPI * r
  jac = TWOPI * SQRT_S

  qkp2  = 0.25d0*Q2 + Q_VAL*k*cos(phi) + k*k
  qkm2  = 0.25d0*Q2 - Q_VAL*k*cos(phi) + k*k
  f_arg = Q2 + 9.0d0 * abs(k*k - 0.25d0*Q2)

  Gp_q2 = exp(-A1*Q2    - A2*Q2*Q2)
  Gp_f  = exp(-A1*f_arg - A2*f_arg*f_arg)

  ! Compute alpha_D for qkp2
  log_num = dlog((qkp2 + RHO_MG2) / LAMBDA2)
  expo = -1.0d0 - GAMMA1
  m2s = MG2 * (log_num / LOG_DEN)**expo
  arg_log_aD = (qkp2 + 4.0d0*m2s) / LAMBDA2
  log_aD = dlog(arg_log_aD)
  denom_aD = B0 * (qkp2 + m2s) * log_aD
  aD_p = 1.0d0 / denom_aD

  ! Compute alpha_D for qkm2
  log_num = dlog((qkm2 + RHO_MG2) / LAMBDA2)
  m2s = MG2 * (log_num / LOG_DEN)**expo
  arg_log_aD = (qkm2 + 4.0d0*m2s) / LAMBDA2
  log_aD = dlog(arg_log_aD)
  denom_aD = B0 * (qkm2 + m2s) * log_aD
  aD_m = 1.0d0 / denom_aD

  T1val = aD_p * aD_m * Gp_q2 * Gp_q2
  T2val = aD_p * aD_m * Gp_f  * (2.0d0*Gp_q2 - Gp_f)

  ff(1) = k * (T1val - T2val) * jac

end subroutine integrand_ID
