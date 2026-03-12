!==============================================================================
! I_1.f90
!
! WHAT THIS PROGRAM DOES:
!   Numerically computes a 2D integral that appears in physics calculations.
!   The integral has two variables:
!     - k   : a radial momentum, ranging from 0 to infinity
!     - phi : an angle, ranging from 0 to 2*pi (a full circle)
!
!   The mathematical formula being evaluated is:
!
!     I1 = INT_0^{2pi} dphi  INT_0^inf dk
!               k * q^2 / [ (k^2 + mu^2) * (k^2 - 2kq*cos(phi) + q^2 + mu^2) ]
!
! HOW IT HANDLES THE INFINITE RANGE OF k:
!   A computer cannot integrate directly to infinity, so the program uses a
!   mathematical trick called a variable substitution:
!
!       k = t / (1 - t),    with t in [0, 1)
!
!   This maps the infinite range [0, +inf) onto the finite interval [0, 1),
!   which the integrator can handle. As t approaches 1, k approaches infinity.
!   The correction factor (Jacobian) introduced by this substitution is:
!
!       dk/dt = 1 / (1 - t)^2
!
!   So Cuba integrates both variables over the unit square [0,1]^2:
!       xx(1) = t  ->  k   = t / (1 - t)     covers k   in [0, +inf)
!       xx(2) = r  ->  phi = 2*pi * r          covers phi in [0, 2*pi]
!
! EXTERNAL LIBRARY USED:
!   Cuba (CUHRE) -- a professional library for multidimensional numerical
!   integration. Download: https://feynarts.de/cuba/
!
! HOW TO COMPILE:
!   gfortran -O2 -o I_1 I_1.f90 -lcuba -lm
!
! HOW TO RUN:
!   Edit the values of MU and Q2 in the "USER INPUT" section of the main
!   program below, then recompile and run:  ./I_1
!==============================================================================


!------------------------------------------------------------------------------
! MODULE: params_mod
!
! WHAT IT IS:
!   A shared storage box that holds two physical numbers (mu and q).
!   Any part of the program can read from this box at any time.
!
! WHY IT EXISTS:
!   The Cuba library calls the integrand subroutine with a fixed, rigid list
!   of arguments -- you are not allowed to add extra arguments to it. The
!   workaround is to store the physical parameters here so the integrand
!   subroutine can silently read them without needing extra arguments.
!
! VARIABLES STORED:
!   mu_val -- the infrared regulator mu (a small positive number that prevents
!             the integral from diverging at low k). Must be > 0.
!   q_val  -- the external momentum q (the magnitude of an external momentum
!             vector in the physics problem).
!
!   Both are real(8), meaning 64-bit double-precision decimal numbers
!   (about 15-16 significant digits of accuracy).
!   The "save" attribute ensures their values persist between subroutine calls.
!------------------------------------------------------------------------------
module params_mod
  implicit none
  real(8), save :: mu_val    ! mu parameter
  real(8), save :: q_val     ! q momentum
end module params_mod


!------------------------------------------------------------------------------
! PROGRAM: integral_I1
!
! WHAT IT DOES:
!   This is the main program -- execution starts here. It:
!     1. Stores the physical parameters (MU, Q) into the shared module.
!     2. Calls the Cuba CUHRE integrator to numerically compute I1.
!     3. Prints a formatted results table to the screen.
!------------------------------------------------------------------------------
program integral_I1
  use params_mod
  implicit none

  !----------------------------------------------------------------------------
  ! USER INPUT -- edit these two values before compiling
  !
  !   MU : the infrared regulator. Must be greater than zero.
  !   Q2 : the external momentum squared (q^2). Q is derived automatically.
  !----------------------------------------------------------------------------
  real(8), parameter :: MU    = 1.0d0    ! infrared regulator (must be > 0)
  real(8), parameter :: Q2    = 5.0d0    ! external momentum squared
  real(8), parameter :: Q     = sqrt(Q2) ! external momentum, derived from Q2

  !----------------------------------------------------------------------------
  ! CUBA CUHRE SETTINGS
  !
  ! These control how the numerical integrator behaves. You usually do not
  ! need to change these unless you want different precision or performance.
  !
  !   NDIM    : number of integration dimensions. 2 here (k and phi).
  !   NCOMP   : number of integrands to compute. 1 here (just I1).
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
  integer, parameter :: NCOMP   = 1          ! number of integrands
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
  !   integral(1) : the numerical result of the integral (the main answer)
  !   error(1)    : Cuba's estimated error on the result
  !   prob(1)     : chi-square probability; close to 1 means the error estimate
  !                 is reliable
  !----------------------------------------------------------------------------
  integer :: nregions           ! number of subregions used during integration
  integer :: neval              ! actual number of integrand evaluations performed
  integer :: fail               ! convergence flag: 0 = converged, 1 = max evals reached

  real(8) :: integral(NCOMP)   ! estimated value of the integral
  real(8) :: error(NCOMP)      ! estimated absolute error on integral
  real(8) :: prob(NCOMP)       ! chi-square probability (reliability of error estimate)

  ! PI constant
  real(8), parameter :: PI = 3.14159265358979323846d0

  ! Declaration of the external integrand subroutine (defined below)
  external :: integrand_I1

  !----------------------------------------------------------------------------
  ! Step 1: Store MU and Q into the shared module so integrand_I1 can read them
  !----------------------------------------------------------------------------
  call set_params(MU, Q)

  !----------------------------------------------------------------------------
  ! Step 2: Call Cuba CUHRE to integrate over the unit square [0,1]^2.
  !
  ! Cuba will call integrand_I1 many times, each time at a different point
  ! (xx(1), xx(2)) in [0,1]^2. Inside the integrand, those points are mapped:
  !   xx(1) = t in [0,1)  ->  k   = t / (1 - t)   covers k in [0, +inf)
  !   xx(2) = r in [0,1]  ->  phi = 2*pi * r        covers phi in [0, 2*pi]
  !----------------------------------------------------------------------------
  call Cuhre(NDIM, NCOMP, integrand_I1, 0,  &  ! dimensions, components, integrand, userdata
             NVEC,                            &  ! points per batch
             EPSREL, EPSABS,                 &  ! error targets
             FLAGS,                           &  ! verbosity
             MINEVAL, MAXEVAL,               &  ! evaluation bounds
             KEY,                             &  ! cubature rule selector
             STATEFILE, SPIN,                &  ! state file, parallelisation
             nregions, neval, fail,          &  ! output: regions, evals, status
             integral, error, prob)             ! output: result, error, chi2 prob

  !----------------------------------------------------------------------------
  ! Step 3: Print results to the screen
  !----------------------------------------------------------------------------
  write(*,*)
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a)') '  |            Cuba CUHRE  --  Result I1            |'
  write(*,'(a)') '  |         (full range, no tail correction)        |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f8.4,a,f8.4,a)') &
       '  |   mu = ', MU, '   q2 = ', Q2, '                       |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f18.10,a)')  '  |   Result        ', integral(1), '           |'
  write(*,'(a,es14.4,a)')  '  |   Est. error    ', error(1),    '               |'
  write(*,'(a,i12,a)')     '  |   Evaluations   ', neval,        '               |'
  write(*,'(a,i12,a)')     '  |   Subregions    ', nregions,     '               |'
  if (fail == 0) then
    write(*,'(a)') '  |   Status        CONVERGED                       |'
  else
    write(*,'(a)') '  |   Status        WARNING: max evals reached      |'
  end if
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,*)

end program integral_I1


!------------------------------------------------------------------------------
! SUBROUTINE: set_params
!
! WHAT IT DOES:
!   A simple one-time setup helper. It takes the two physical values (mu and q)
!   from the main program and writes them into the shared module params_mod,
!   making them available to integrand_I1 later.
!
! ARGUMENTS:
!   mu  [input, real(8)] -- the infrared regulator value to be stored.
!                           "intent(in)" means this subroutine only reads it,
!                           never modifies it.
!   q   [input, real(8)] -- the external momentum value to be stored.
!                           "intent(in)" means this subroutine only reads it,
!                           never modifies it.
!
! WHAT HAPPENS INSIDE:
!   mu_val = mu   <- writes mu into the shared module
!   q_val  = q    <- writes q  into the shared module
!------------------------------------------------------------------------------
subroutine set_params(mu, q)
  use params_mod
  implicit none
  real(8), intent(in) :: mu    
  real(8), intent(in) :: q     
  mu_val = mu
  q_val  = q
end subroutine set_params


!------------------------------------------------------------------------------
! SUBROUTINE: integrand_I1
!
! WHAT IT DOES:
!   This is the heart of the calculation. Cuba calls this subroutine thousands
!   (or millions) of times. Each call provides a point (xx(1), xx(2)) in the
!   unit square [0,1]^2, and this subroutine must return the value of the
!   integrand at that point via ff(1).
!
! IMPORTANT -- FIXED INTERFACE:
!   The argument list below is dictated by the Cuba library. You cannot add,
!   remove, or rename any argument. Physical parameters are accessed through
!   the shared module params_mod instead.
!
! ARGUMENTS:
!   ndim        [input,  integer     ] -- number of integration dimensions.
!                                         Always 2 here (k and phi). Set by Cuba.
!   xx(ndim)    [input,  real(8) array] -- the integration point in [0,1]^2.
!                                         xx(1) maps to k; xx(2) maps to phi.
!                                         Provided by Cuba each call.
!   ncomp       [input,  integer     ] -- number of integrand components.
!                                         Always 1 here. Set by Cuba.
!   ff(ncomp)   [OUTPUT, real(8) array] -- this subroutine must fill ff(1) with
!                                         the integrand value at the given point.
!                                         Cuba reads this after each call.
!   userdata    [input,  integer     ] -- an optional extra integer passed from
!                                         the Cuba call. Not used here (set to 0).
!
! INTERNAL VARIABLES:
!   t      -- the raw value of xx(1); the mapped variable for k; lives in [0,1)
!   k      -- the actual momentum, recovered via k = t / (1 - t); in [0, +inf)
!   phi    -- the azimuthal angle, recovered via phi = 2*pi * xx(2); in [0, 2*pi]
!   q2     -- precomputed q^2 to avoid redundant multiplications
!   denom1 -- first  denominator of the integrand: k^2 + mu^2
!   denom2 -- second denominator of the integrand: k^2 - 2kq*cos(phi) + q^2 + mu^2
!   jac    -- Jacobian of the variable substitution: 2*pi / (1-t)^2
!             This corrects the integrand for the change of variables from
!             (k, phi) to (t, r). Without it the answer would be wrong.
!   TWOPI  -- the constant 2*pi, defined at full double precision
!
! STEP-BY-STEP LOGIC:
!   1. If t = 1, k would be infinite -> return 0 immediately (safe boundary).
!   2. Recover k   = t / (1 - t)
!   3. Recover phi = 2*pi * xx(2)
!   4. Compute Jacobian: jac = 2*pi / (1 - t)^2
!   5. Evaluate denom1 and denom2
!   6. Safety check: if either denominator is essentially zero, return 0.
!   7. Return ff(1) = (k * q^2 / (denom1 * denom2)) * jac
!------------------------------------------------------------------------------
subroutine integrand_I1(ndim, xx, ncomp, ff, userdata)
  use params_mod
  implicit none

  integer, intent(in)  :: ndim          ! number of dimensions (fixed by Cuba)
  real(8), intent(in)  :: xx(ndim)      ! integration point in [0,1]^2
  integer, intent(in)  :: ncomp         ! number of integrand components (fixed by Cuba)
  real(8), intent(out) :: ff(ncomp)     ! integrand value to be filled
  integer, intent(in)  :: userdata      ! unused extra argument required by Cuba interface

  real(8), parameter :: TWOPI = 6.28318530717958647692d0

  real(8) :: t       ! mapped variable for k: t in [0, 1)
  real(8) :: k       ! integration variable: radial momentum, k in [0, +inf)
  real(8) :: phi     ! integration variable: azimuthal angle, phi in [0, 2*pi]
  real(8) :: q2      ! q^2, precomputed to avoid redundant multiplication
  real(8) :: denom1  ! first  denominator:  k^2 + mu^2
  real(8) :: denom2  ! second denominator:  k^2 - 2kq*cos(phi) + q^2 + mu^2
  real(8) :: jac     ! Jacobian: (1/(1-t)^2) * 2*pi

  t = xx(1)

  ! At the boundary t = 1, k diverges; integrand * J -> 0, return 0 explicitly
  if (t >= 1d0) then
    ff(1) = 0d0
    return
  end if

  ! Variable substitution: k = t / (1 - t)
  k = t / (1d0 - t)

  ! Azimuthal angle: phi = 2*pi * r
  phi = TWOPI * xx(2)

  ! Precompute q^2
  q2 = q_val * q_val

  ! Jacobian: dk/dt * dphi/dr = 1/(1-t)^2 * 2*pi
  jac = TWOPI / (1d0 - t)**2

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

end subroutine integrand_I1
