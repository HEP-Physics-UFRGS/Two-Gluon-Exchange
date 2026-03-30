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
  real(8), parameter :: Q2     = 1.0d0       ! external momentum squared [GeV^2]
  real(8), parameter :: MG     = 0.380d0       ! gluon mass parameter mg [GeV]
  real(8), parameter :: LAMBDA = 0.284d0      ! QCD scale Lambda [GeV]
  real(8), parameter :: RHO    = 4.0d0       ! IR regulator rho in m2log
  real(8), parameter :: GAMMA1 = 0.084d0       ! exponent gamma1 in m2log
  real(8), parameter :: B0     = 0.71619724391D0   ! beta-function coeff b0
  real(8), parameter :: A1     = 1.491d0       ! Gp first coeff  [GeV^-2]
  real(8), parameter :: A2     = 2.77d0      ! Gp second coeff [GeV^-4]
  ! --------------------------------------------------------------------------

  ! Derived (do not edit)
  real(8), parameter :: Q_VAL  = sqrt(Q2)    ! |q| = sqrt(q2)

end module params_mod


!=============================================================================
! PROGRAM integral_ID
!=============================================================================
program integral_ID
  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
  use params_mod
  implicit none
  
  ! Cuba SUAVE settings
  integer,   parameter :: NDIM     = 2
  integer,   parameter :: NCOMP    = 1
  integer,   parameter :: NVEC     = 1
  real(8),   parameter :: EPSREL   = 1.0d-8
  real(8),   parameter :: EPSABS   = 1.0d-16
  integer,   parameter :: FLAGS    = 0
  integer,   parameter :: MINEVAL  = 100000
  integer,   parameter :: MAXEVAL  = 20000000
  integer,   parameter :: NNEW     = 10000
  integer,   parameter :: NMIN     = 2
  real(8),   parameter :: FLATNESS = 25.0d0
  integer,   parameter :: SPIN     = -1
  
  ! Cuba output
  integer :: nregions, neval, fail
  real(8) :: integral(NCOMP), error(NCOMP), prob(NCOMP)
  
  ! Declare integrand as external
  external :: integrand_ID
  
  type(c_ptr) :: userdata = c_null_ptr
  
  ! Call Cuba SUAVE
  call suave(NDIM, NCOMP, integrand_ID, userdata, NVEC, &
             EPSREL, EPSABS, FLAGS, &
             MINEVAL, MAXEVAL, &
             NNEW, NMIN, FLATNESS, &
             '', SPIN, &
             nregions, neval, fail, &
             integral, error, prob)
  
  ! Print results
  write(*,*)
  write(*,'(a)') '  +----------------------------------------------------------+'
  write(*,'(a)') '  |           Cuba SUAVE  --  Result  I_D                    |'
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

contains

!=============================================================================
! SUBROUTINE integrand_ID
! Called by Cuba at each integration point (xx(1), xx(2)) in [0,1]^2.
!=============================================================================
subroutine integrand_ID(ndim, xx, ncomp, ff, userdata)
  use, intrinsic :: iso_c_binding, only: c_ptr
  use params_mod
  implicit none

  integer, intent(in)  :: ndim
  real(8), intent(in)  :: xx(ndim)
  integer, intent(in)  :: ncomp
  real(8), intent(out) :: ff(ncomp)
  type(c_ptr), intent(in), value :: userdata

  real(8), parameter :: TWOPI = 6.28318530717958647692d0

  real(8) :: t, r, k, phi, jac
  real(8) :: qkp2, qkm2, f_arg
  real(8) :: aD_p, aD_m
  real(8) :: Gp_q2, Gp_f
  real(8) :: T1val, T2val

  ! Avoid unused parameter warning
  if (.false.) then
    print *, userdata
  end if

  t = xx(1)
  r = xx(2)

  if (t >= 1.0d0) then
    ff(1) = 0.0d0
    return
  end if

  k   = t / (1.0d0 - t)
  phi = TWOPI * r
  jac = TWOPI / (1.0d0 - t)**2

  qkp2  = 0.25d0*Q2 + Q_VAL*k*cos(phi) + k*k
  qkm2  = 0.25d0*Q2 - Q_VAL*k*cos(phi) + k*k
  f_arg = Q2 + 9.0d0 * abs(k*k - 0.25d0*Q2)

  Gp_q2 = exp(-A1*Q2    - A2*Q2*Q2)
  Gp_f  = exp(-A1*f_arg - A2*f_arg*f_arg)

  call eval_alpha_D(qkp2, MG, LAMBDA, RHO, GAMMA1, B0, aD_p)
  call eval_alpha_D(qkm2, MG, LAMBDA, RHO, GAMMA1, B0, aD_m)

  T1val = aD_p * aD_m * Gp_q2 * Gp_q2
  T2val = aD_p * aD_m * Gp_f  * (2.0d0*Gp_q2 - Gp_f)

  ff(1) = k * (T1val - T2val) * jac

end subroutine integrand_ID

end program integral_ID


!=============================================================================
! SUBROUTINE eval_alpha_D
!
! Computes alpha_D(s) = 1 / { b0 * (s + m2log(s)) * ln[(s+4*m2log(s))/L^2] }
! with m2log(s) = mg^2 * { ln[(s+rho*mg^2)/L^2] / ln[rho*mg^2/L^2] }^(-1-g1)
!
! Returns alpha = 0 on any numerical safety failure.
!=============================================================================
subroutine eval_alpha_D(s, mg, lam, rho, gam1, b0, alpha)
  implicit none

  real(8), intent(in)  :: s, mg, lam, rho, gam1, b0
  real(8), intent(out) :: alpha

  real(8), parameter :: TINY = 1.0d-30

  real(8) :: mg2, lam2, rho_mg2
  real(8) :: log_num, log_den, expo
  real(8) :: m2s, arg_log_aD, log_aD, denom_aD

  alpha   = 0.0d0
  mg2     = mg * mg
  lam2    = lam * lam
  rho_mg2 = rho * mg2

  if (rho_mg2 <= 0.0d0 .or. lam2 <= 0.0d0) return
  log_den = log(rho_mg2 / lam2)
  if (abs(log_den) < TINY) return

  if ((s + rho_mg2) <= 0.0d0) return
  log_num = log((s + rho_mg2) / lam2)

  expo = -1.0d0 - gam1
  m2s  = mg2 * (log_num / log_den)**expo
  if (m2s < 0.0d0) return

  arg_log_aD = (s + 4.0d0*m2s) / lam2
  if (arg_log_aD <= 0.0d0) return
  log_aD = log(arg_log_aD)
  if (abs(log_aD) < TINY) return

  denom_aD = b0 * (s + m2s) * log_aD
  if (abs(denom_aD) < TINY) return

  alpha = 1.0d0 / denom_aD

end subroutine eval_alpha_D
