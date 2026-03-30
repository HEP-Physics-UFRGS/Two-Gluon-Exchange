!==============================================================================
! I_2.f90
!
! WHAT THIS PROGRAM DOES:
!   Numerically computes a 2D integral that appears in physics calculations.
!   The integral has two variables:
!     - k   : a radial momentum, ranging from 0 to a finite cutoff K_CUT = 10
!     - phi : an angle, ranging from 0 to 2*pi (a full circle)
!
!   The mathematical formula being evaluated is:
!
!     I2 = INT_0^{2pi} dphi  INT_0^{10} dk
!               k * q^2 / [ (k^2 + mu^2) * (k^2 - 2kq*cos(phi) + q^2 + mu^2) ]
!
! HOW IT DIFFERS FROM I_1.f90:
!   In I_1.f90, k ranged from 0 to INFINITY, which required a special variable
!   substitution trick (k = t/(1-t)) to map infinity onto a finite interval.
!
!   Here, k only goes up to a finite upper limit K_CUT = 10, so NO such trick
!   is needed. Cuba integrates directly using a simple linear rescaling:
!
!       k   = K_CUT * s,    s in [0, 1]    (stretches [0,1] to [0, K_CUT])
!       phi = 2*pi  * r,    r in [0, 1]    (stretches [0,1] to [0, 2*pi])
!
!   The correction factor (Jacobian) for this linear mapping is simply:
!       J = K_CUT * 2*pi
!
! EXTERNAL LIBRARY USED:
!   Cuba (CUHRE) -- a professional library for multidimensional numerical
!   integration. Download: https://feynarts.de/cuba/
!
! HOW TO COMPILE:
!   gfortran -O2 -o integral_I2 integral_I2.f90 -lcuba -lm
!
! HOW TO RUN:
!   Edit the values of MU and Q2 in the "USER INPUT" section of the main
!   program below, then recompile and run:  ./integral_I2
!==============================================================================


!------------------------------------------------------------------------------
! MODULE: params_mod
!
! WHAT IT IS:
!   A shared storage box that holds three numbers the rest of the program needs.
!   Any part of the program can read from this box at any time.
!
! WHY IT EXISTS:
!   The Cuba library calls the integrand subroutine with a fixed, rigid list
!   of arguments -- you are not allowed to add extra arguments to it. The
!   workaround is to store the physical parameters here so the integrand
!   subroutine can silently read them without needing extra arguments.
!
! VARIABLES STORED:
!   mu_val   -- the infrared regulator mu. A small positive number that prevents
!               the integral from diverging. Must be > 0.
!   q_val    -- the external momentum q. The magnitude of an external momentum
!               vector in the physics problem.
!   kcut_val -- the upper limit of integration for k. Set to 10 in this program.
!               Stored here so the integrand can use it for the variable mapping.
!
!   All three are real(8): 64-bit double-precision decimal numbers (~15-16
!   significant digits). The "save" attribute ensures values are not reset
!   between subroutine calls.
!------------------------------------------------------------------------------
module params_mod
  implicit none
  real(8), save :: mu_val    ! infrared regulator / gluon mass parameter
  real(8), save :: q_val     ! external momentum
  real(8), save :: kcut_val  ! upper integration limit passed to the integrand
end module params_mod


!------------------------------------------------------------------------------
! PROGRAM: integral_I2
!
! WHAT IT DOES:
!   This is the main program -- execution starts here. It:
!     1. Stores the physical parameters (MU, Q, K_CUT) into the shared module.
!     2. Calls the Cuba CUHRE integrator to numerically compute I2.
!     3. Prints a formatted results table to the screen.
!------------------------------------------------------------------------------
program integral_I2
  use params_mod
  implicit none

  !----------------------------------------------------------------------------
  ! USER INPUT -- edit these values before compiling
  !
  !   MU    : the infrared regulator. Must be greater than zero.
  !   Q2    : the external momentum squared (q^2). Q is derived automatically.
  !   K_CUT : the upper limit of the k integration. Set to 10 here because
  !           the integral converges well within this range. Increase if needed.
  !----------------------------------------------------------------------------
  real(8), parameter :: MU    = 1.0d0    ! infrared regulator (must be > 0)
  real(8), parameter :: Q2    = 4.0d0    ! external momentum squared (set 0 for trivial case)
  real(8), parameter :: Q     = sqrt(Q2) ! external momentum, derived from Q2

  ! Upper cutoff for the numerical integration.
  ! Set to 10 as the integral has a finite upper limit.
  real(8), parameter :: K_CUT = 10.0d0

  !----------------------------------------------------------------------------
  ! CUBA CUHRE SETTINGS
  !
  ! These control how the numerical integrator behaves. You usually do not
  ! need to change these unless you want different precision or performance.
  !
  !   NDIM    : number of integration dimensions. 2 here (k and phi).
  !   NCOMP   : number of integrands to compute. 1 here (just I2).
  !   NVEC    : how many points Cuba sends to the integrand per call. 1 here.
  !   EPSREL  : relative error target. Cuba stops when the result is accurate
  !             to ~8 decimal places relative to its size. (1e-8 = 0.000001%)
  !   EPSABS  : absolute error target. A fallback stopping criterion based on
  !             the raw magnitude of the error. (1e-12 is very tight)
  !   FLAGS   : verbosity. 0 = Cuba prints nothing to the screen.
  !   MINEVAL : minimum number of integrand evaluations before Cuba can stop.
  !   MAXEVAL : maximum allowed evaluations. A safety cap to prevent runaway.
  !   KEY     : which cubature rule Cuba uses. 0 = Cuba chooses automatically.
  !   STATEFILE: file to save/restore integration state. Blank = disabled.
  !   SPIN    : parallelisation handle. -1 = no parallel execution.
  !----------------------------------------------------------------------------
  integer, parameter :: NDIM    = 2          ! number of integration dimensions (k, phi)
  integer, parameter :: NCOMP   = 1          ! number of integrands (we have just one: I2)
  integer, parameter :: NVEC    = 1          ! number of points passed to integrand per call
  real(8), parameter :: EPSREL  = 1d-8       ! desired relative error on the result
  real(8), parameter :: EPSABS  = 1d-12      ! desired absolute error on the result
  integer, parameter :: FLAGS   = 0          ! verbosity flag: 0 = silent
  integer, parameter :: MINEVAL = 0          ! minimum number of integrand evaluations
  integer, parameter :: MAXEVAL = 10000000   ! maximum number of integrand evaluations
  integer, parameter :: KEY     = 0          ! cubature rule: 0 = Cuba chooses automatically

  ! STATEFILE: file to save/restore integration state (' ' = disabled)
  character(len=1), parameter :: STATEFILE = ' '

  ! SPIN: parallelisation handle (-1 = not used)
  integer :: SPIN = -1

  !----------------------------------------------------------------------------
  ! OUTPUT VARIABLES -- filled in by Cuba after integration finishes
  !
  !   nregions    : number of subregions Cuba split the domain into for accuracy
  !   neval       : actual number of times the integrand was evaluated
  !   fail        : convergence flag. 0 = success. 1 = hit MAXEVAL, not converged.
  !   integral(1) : the numerical result of the finite integral over [0, K_CUT]
  !   error(1)    : Cuba's estimated error on the result
  !   prob(1)     : chi-square probability; close to 1 means the error estimate
  !                 is reliable
  !   I2          : the final answer (here it equals integral(1) directly,
  !                 since no tail correction is needed for a finite upper limit)
  !----------------------------------------------------------------------------
  integer :: nregions           ! number of subregions used during integration
  integer :: neval              ! actual number of integrand evaluations performed
  integer :: fail               ! convergence flag: 0 = converged, 1 = max evals reached

  real(8) :: integral(NCOMP)   ! estimated value of the finite part [0, K_CUT]
  real(8) :: error(NCOMP)      ! estimated absolute error on integral
  real(8) :: prob(NCOMP)       ! chi-square probability (reliability of error estimate)
  real(8) :: I2                 ! final result

  ! PI constant
  real(8), parameter :: PI = 3.14159265358979323846d0

  ! Declaration of the external integrand subroutine (defined below)
  external :: integrand_I2

  !----------------------------------------------------------------------------
  ! Step 1: Store MU, Q, and K_CUT into the shared module
  !----------------------------------------------------------------------------
  call set_params(MU, Q, K_CUT)

  !----------------------------------------------------------------------------
  ! Step 2: Call Cuba CUHRE to integrate over the unit square [0,1]^2.
  !
  ! Cuba will call integrand_I2 many times, each time at a different point
  ! (xx(1), xx(2)) in [0,1]^2. Inside the integrand, those points are mapped:
  !   xx(1) = s in [0,1]  ->  k   = K_CUT * s    covers k   in [0, K_CUT]
  !   xx(2) = r in [0,1]  ->  phi = 2*pi * r       covers phi in [0, 2*pi]
  !----------------------------------------------------------------------------
  call Cuhre(NDIM, NCOMP, integrand_I2, 0,  &  ! dimensions, components, integrand, userdata
             NVEC,                            &  ! points per batch
             EPSREL, EPSABS,                 &  ! error targets
             FLAGS,                           &  ! verbosity
             MINEVAL, MAXEVAL,               &  ! evaluation bounds
             KEY,                             &  ! cubature rule selector
             STATEFILE, SPIN,                &  ! state file, parallelisation
             nregions, neval, fail,          &  ! output: regions, evals, status
             integral, error, prob)             ! output: result, error, chi2 prob

  ! Final result (no tail correction needed for finite upper limit)
  I2 = integral(1)

  !----------------------------------------------------------------------------
  ! Step 3: Print results to the screen
  !----------------------------------------------------------------------------
  write(*,*)
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a)') '  |            Cuba CUHRE  --  Result I2            |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f8.4,a,f8.4,a)') &
       '  |   mu = ', MU, '   q2 = ', Q2, '                       |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f18.10,a)')  '  |   I_finite      ', integral(1), '           |'
  write(*,'(a,f18.10,a)')  '  |   Result        ', I2,          '           |'
  write(*,'(a,es14.4,a)')  '  |   Est. error    ', error(1),    '               |'
  write(*,'(a,I22,a)')     '  |   Evaluations   ', neval,        '               |'
  write(*,'(a,I22,a)')     '  |   Subregions    ', nregions,     '               |'
  if (fail == 0) then
    write(*,'(a)') '  |   Status        CONVERGED                       |'
  else
    write(*,'(a)') '  |   Status        WARNING: max evals reached      |'
  end if
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,*)

end program integral_I2


!------------------------------------------------------------------------------
! SUBROUTINE: set_params
!
! WHAT IT DOES:
!   A simple one-time setup helper. It takes the three values (mu, q, kcut)
!   from the main program and writes them into the shared module params_mod,
!   making them available to integrand_I2 later.
!
! ARGUMENTS:
!   mu   [input, real(8)] -- the infrared regulator value to be stored.
!                            "intent(in)" means this subroutine only reads it,
!                            never modifies it.
!   q    [input, real(8)] -- the external momentum value to be stored.
!                            "intent(in)" means this subroutine only reads it,
!                            never modifies it.
!   kcut [input, real(8)] -- the upper k integration limit to be stored.
!                            "intent(in)" means this subroutine only reads it,
!                            never modifies it.
!
! WHAT HAPPENS INSIDE:
!   mu_val   = mu     <- writes mu   into the shared module
!   q_val    = q      <- writes q    into the shared module
!   kcut_val = kcut   <- writes kcut into the shared module
!------------------------------------------------------------------------------
subroutine set_params(mu, q, kcut)
  use params_mod
  implicit none
  real(8), intent(in) :: mu    ! infrared regulator
  real(8), intent(in) :: q     ! external momentum
  real(8), intent(in) :: kcut  ! upper cutoff for numerical integration
  mu_val   = mu
  q_val    = q
  kcut_val = kcut
end subroutine set_params


!------------------------------------------------------------------------------
! SUBROUTINE: integrand_I2
!
! WHAT IT DOES:
!   This is the heart of the calculation. Cuba calls this subroutine thousands
!   (or millions) of times. Each call provides a point (xx(1), xx(2)) in the
!   unit square [0,1]^2, and this subroutine must return the value of the
!   integrand at that point via ff(1).
!
! KEY DIFFERENCE FROM I_1.f90:
!   Because k has a finite upper limit (K_CUT), no special substitution is
!   needed to handle infinity. The mapping here is simply linear:
!       k   = K_CUT * xx(1)     (stretches [0,1] to [0, K_CUT])
!       phi = 2*pi  * xx(2)     (stretches [0,1] to [0, 2*pi])
!   There is no boundary singularity to handle, and no t = 1 edge case.
!
! IMPORTANT -- FIXED INTERFACE:
!   The argument list below is dictated by the Cuba library. You cannot add,
!   remove, or rename any argument. Physical parameters are accessed through
!   the shared module params_mod instead.
!
! ARGUMENTS:
!   ndim        [input,  integer      ] -- number of integration dimensions.
!                                          Always 2 here (k and phi). Set by Cuba.
!   xx(ndim)    [input,  real(8) array] -- the integration point in [0,1]^2.
!                                          xx(1) maps to k; xx(2) maps to phi.
!                                          Provided by Cuba each call.
!   ncomp       [input,  integer      ] -- number of integrand components.
!                                          Always 1 here. Set by Cuba.
!   ff(ncomp)   [OUTPUT, real(8) array] -- this subroutine must fill ff(1) with
!                                          the integrand value at the given point.
!                                          Cuba reads this after each call.
!   userdata    [input,  integer      ] -- an optional extra integer passed from
!                                          the Cuba call. Not used here (set to 0).
!
! INTERNAL VARIABLES:
!   k      -- the actual momentum, recovered via k = K_CUT * xx(1); in [0, K_CUT]
!   phi    -- the azimuthal angle, recovered via phi = 2*pi * xx(2); in [0, 2*pi]
!   q2     -- precomputed q^2 to avoid redundant multiplications
!   denom1 -- first  denominator of the integrand: k^2 + mu^2
!   denom2 -- second denominator of the integrand: k^2 - 2kq*cos(phi) + q^2 + mu^2
!   jac    -- Jacobian of the variable mapping: K_CUT * 2*pi
!             This corrects the integrand for the linear rescaling of both
!             variables. Without it the answer would be wrong.
!   TWOPI  -- the constant 2*pi, defined at full double precision
!
! STEP-BY-STEP LOGIC:
!   1. Recover k   = K_CUT * xx(1)     (linear map from [0,1] to [0, K_CUT])
!   2. Recover phi = 2*pi  * xx(2)     (linear map from [0,1] to [0, 2*pi])
!   3. Compute Jacobian: jac = K_CUT * 2*pi
!   4. Evaluate denom1 and denom2
!   5. Safety check: if either denominator is essentially zero, return 0.
!   6. Return ff(1) = (k * q^2 / (denom1 * denom2)) * jac
!------------------------------------------------------------------------------
subroutine integrand_I2(ndim, xx, ncomp, ff, userdata)
  use params_mod
  implicit none

  integer, intent(in)  :: ndim          ! number of dimensions (fixed by Cuba)
  real(8), intent(in)  :: xx(ndim)      ! integration point in [0,1]^2
  integer, intent(in)  :: ncomp         ! number of integrand components (fixed by Cuba)
  real(8), intent(out) :: ff(ncomp)     ! integrand value to be filled
  integer, intent(in)  :: userdata      ! unused extra argument required by Cuba interface

  ! 2*pi as a constant (phi integration range)
  real(8), parameter :: TWOPI = 6.28318530717958647692d0

  real(8) :: k       ! integration variable: radial momentum, k in [0, K_CUT]
  real(8) :: phi     ! integration variable: azimuthal angle, phi in [0, 2*pi]
  real(8) :: q2      ! q^2, precomputed to avoid redundant multiplication
  real(8) :: denom1  ! first  denominator:  k^2 + mu^2
  real(8) :: denom2  ! second denominator:  k^2 - 2kq*cos(phi) + q^2 + mu^2
  real(8) :: jac     ! Jacobian of the variable mapping: K_CUT * 2*pi

  ! Linear mapping: s -> k = K_CUT * s
  k   = kcut_val * xx(1)

  ! Linear mapping: r -> phi = 2*pi * r
  phi = TWOPI * xx(2)

  ! Precompute q^2
  q2 = q_val * q_val

  ! Jacobian of both mappings combined
  jac = kcut_val * TWOPI

  ! Evaluate the two denominators of the physical integrand
  denom1 = k*k + mu_val*mu_val
  denom2 = k*k - 2d0*k*q_val*cos(phi) + q2 + mu_val*mu_val

  ! Safety check: avoid division by zero (should not occur for mu > 0)
  if (abs(denom1) < 1d-30 .or. abs(denom2) < 1d-30) then
    ff(1) = 0d0
    return
  end if

  ! Physical integrand times Jacobian
  ff(1) = (k * q2 / (denom1 * denom2)) * jac

end subroutine integrand_I2
