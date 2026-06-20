!==============================================================================
! I_3.f90
!
! WHAT THIS PROGRAM DOES:
!   Numerically computes a 3D integral (three nested integrals) that appears
!   in physics calculations. The integral has THREE variables:
!     - k   : a radial momentum, ranging from 0 to INFINITY
!     - phi : an angle, ranging from 0 to 2*pi (a full circle)
!     - q   : an external momentum, ranging from 0 to Q_MAX (finite)
!
!   The mathematical formula being evaluated is:
!
!     I3 = INT_0^{Q_MAX} dq  INT_0^{2pi} dphi  INT_0^inf dk
!               k * q^2 / [ (k^2 + mu^2) * (k^2 - 2kq*cos(phi) + q^2 + mu^2) ]
!
! HOW IT DIFFERS FROM I_1.f90 AND I_2.f90:
!   - I_1.f90: 2D integral (k and phi only), k goes to infinity
!   - I_2.f90: 2D integral (k and phi only), k has a finite cutoff
!   - I_3.f90: 3D integral -- adds a third integration over q from 0 to Q_MAX.
!              k still goes to infinity and still needs the substitution trick.
!
! HOW IT HANDLES THE INFINITE RANGE OF k:
!   A computer cannot integrate directly to infinity, so the program uses a
!   variable substitution to map the infinite range onto [0, 1):
!
!       k = t / (1 - t),    t in [0, 1)
!
!   As t approaches 1, k approaches infinity. The correction factor (Jacobian)
!   introduced by this substitution is:
!       dk/dt = 1 / (1 - t)^2
!
!   The other two variables use simple linear mappings:
!       phi = 2*pi * r,    r in [0, 1]
!       q   = Q_MAX * s,   s in [0, 1]
!
!   So Cuba integrates all three variables over the unit cube [0,1]^3:
!       xx(1) = t  ->  k   = t / (1 - t)     covers k   in [0, +inf)
!       xx(2) = r  ->  phi = 2*pi * r          covers phi in [0, 2*pi]
!       xx(3) = s  ->  q   = Q_MAX * s         covers q   in [0, Q_MAX]
!
!   The combined Jacobian for all three mappings is:
!       J = (1 / (1-t)^2) * 2*pi * Q_MAX
!
! EXTERNAL LIBRARY USED:
!   Cuba (CUHRE) -- a professional library for multidimensional numerical
!   integration. Download: https://feynarts.de/cuba/
!
! HOW TO COMPILE:
!   gfortran -O2 -o I_3_no_tail I_3_no_tail.f90 -lcuba -lm
!
! HOW TO RUN:
!   Edit the values of MU and Q_MAX in the "USER INPUT" section of the main
!   program below, then recompile and run:  ./I_3_no_tail
!==============================================================================


!------------------------------------------------------------------------------
! MODULE: params_mod
!
! WHAT IT IS:
!   A shared storage box that holds one physical number (mu) that the rest of
!   the program needs. Any part of the program can read from this box.
!
! WHY IT EXISTS:
!   The Cuba library calls the integrand subroutine with a fixed, rigid list
!   of arguments -- you are not allowed to add extra arguments to it. The
!   workaround is to store the physical parameter here so the integrand
!   subroutine can silently read it without needing an extra argument.
!
! NOTE ON Q_MAX:
!   Unlike I_2.f90, Q_MAX is NOT stored in this module. Instead it is defined
!   as a local constant inside the integrand subroutine directly (it must match
!   the value set in the main program).
!
! VARIABLE STORED:
!   mu_val -- the infrared regulator mu. A small positive number that prevents
!             the integral from diverging. Must be > 0.
!
!   real(8) means 64-bit double-precision decimal (~15-16 significant digits).
!   The "save" attribute ensures the value persists between subroutine calls.
!------------------------------------------------------------------------------
module params_mod
  implicit none
  real(8), save :: mu_val    ! infrared regulator / gluon mass parameter
end module params_mod


!------------------------------------------------------------------------------
! PROGRAM: integral_I3
!
! WHAT IT DOES:
!   This is the main program -- execution starts here. It:
!     1. Stores the physical parameter (MU) into the shared module.
!     2. Calls the Cuba CUHRE integrator to numerically compute I3.
!     3. Prints a formatted results table to the screen.
!------------------------------------------------------------------------------
program integral_I3
  use params_mod
  implicit none

  !----------------------------------------------------------------------------
  ! USER INPUT -- edit these values before compiling
  !
  !   MU    : the infrared regulator. Must be greater than zero.
  !   Q_MAX : the upper limit of the q integration. Increase to integrate
  !           over a wider range of external momenta.
  !
  ! IMPORTANT: if you change Q_MAX here, you must also change the Q_MAX
  ! parameter inside the integrand_I3 subroutine below to match.
  !----------------------------------------------------------------------------
  real(8), parameter :: MU    = 1.0d0    ! infrared regulator (must be > 0)
  real(8), parameter :: Q_MAX = 5.0d0    ! upper limit of q integration

  !----------------------------------------------------------------------------
  ! CUBA CUHRE SETTINGS
  !
  ! These control how the numerical integrator behaves. You usually do not
  ! need to change these unless you want different precision or performance.
  !
  !   NDIM    : number of integration dimensions. NOW 3 (k, phi, AND q).
  !   NCOMP   : number of integrands to compute. 1 here (just I3).
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
  integer, parameter :: NDIM    = 3          ! integration dimensions: k, phi, q
  integer, parameter :: NCOMP   = 1          ! number of integrands
  integer, parameter :: NVEC    = 1          ! number of points passed to integrand per call
  real(8), parameter :: EPSREL  = 1d-10       ! desired relative error on the result
  real(8), parameter :: EPSABS  = 1d-16      ! desired absolute error on the result
  integer, parameter :: FLAGS   = 0          ! verbosity flag: 0 = silent
  integer, parameter :: MINEVAL = 1000000          ! minimum number of integrand evaluations
  integer, parameter :: MAXEVAL = 1000000000   ! maximum number of integrand evaluations
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
  external :: integrand_I3

  !----------------------------------------------------------------------------
  ! Step 1: Store MU into the shared module so integrand_I3 can read it
  !----------------------------------------------------------------------------
  call set_params(MU)

  !----------------------------------------------------------------------------
  ! Step 2: Call Cuba CUHRE to integrate over the unit cube [0,1]^3.
  !
  ! Cuba will call integrand_I3 many times, each time at a different point
  ! (xx(1), xx(2), xx(3)) in [0,1]^3. Inside the integrand, those points are
  ! mapped:
  !   xx(1) = t in [0,1)  ->  k   = t / (1-t)    covers k   in [0, +inf)
  !   xx(2) = r in [0,1]  ->  phi = 2*pi * r       covers phi in [0, 2*pi]
  !   xx(3) = s in [0,1]  ->  q   = Q_MAX * s      covers q   in [0, Q_MAX]
  !----------------------------------------------------------------------------
  call Cuhre(NDIM, NCOMP, integrand_I3, 0,  &  ! dimensions, components, integrand, userdata
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
  write(*,'(a)') '  |            Cuba CUHRE  --  Result I3            |'
  write(*,'(a)') '  |         (full range, no tail correction)        |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f8.4,a,f8.4,a)') &
       '  |   mu = ', MU, '   q in [0, ', Q_MAX, ']              |'
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

end program integral_I3


!------------------------------------------------------------------------------
! SUBROUTINE: set_params
!
! WHAT IT DOES:
!   A simple one-time setup helper. It takes mu from the main program and
!   writes it into the shared module params_mod, making it available to
!   integrand_I3 later.
!
! ARGUMENTS:
!   mu  [input, real(8)] -- the infrared regulator value to be stored.
!                           "intent(in)" means this subroutine only reads it,
!                           never modifies it.
!
! WHAT HAPPENS INSIDE:
!   mu_val = mu    <- writes mu into the shared module
!------------------------------------------------------------------------------
subroutine set_params(mu)
  use params_mod
  implicit none
  real(8), intent(in) :: mu    ! infrared regulator
  mu_val = mu
end subroutine set_params


!------------------------------------------------------------------------------
! SUBROUTINE: integrand_I3
!
! WHAT IT DOES:
!   This is the heart of the calculation. Cuba calls this subroutine thousands
!   (or millions) of times. Each call provides a point (xx(1), xx(2), xx(3))
!   in the unit cube [0,1]^3, and this subroutine must return the value of the
!   integrand at that point via ff(1).
!
! KEY DIFFERENCE FROM I_1.f90:
!   This is a 3D version -- there is now a third integration variable q,
!   which is mapped linearly: q = Q_MAX * xx(3). The k variable still uses
!   the infinity-handling substitution k = t/(1-t), just like I_1.f90.
!
! IMPORTANT -- FIXED INTERFACE:
!   The argument list below is dictated by the Cuba library. You cannot add,
!   remove, or rename any argument. Physical parameters are accessed through
!   the shared module params_mod instead.
!
! IMPORTANT -- Q_MAX DUPLICATION:
!   Q_MAX is defined as a local constant inside this subroutine. It must always
!   be kept equal to the Q_MAX defined in the main program above. If you change
!   one, change the other too.
!
! ARGUMENTS:
!   ndim        [input,  integer      ] -- number of integration dimensions.
!                                          Always 3 here (k, phi, q). Set by Cuba.
!   xx(ndim)    [input,  real(8) array] -- the integration point in [0,1]^3.
!                                          xx(1)->k, xx(2)->phi, xx(3)->q.
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
!   t      -- the raw value of xx(1); the mapped variable for k; lives in [0,1)
!   k      -- the actual momentum, recovered via k = t/(1-t); in [0, +inf)
!   phi    -- the azimuthal angle, recovered via phi = 2*pi*xx(2); in [0, 2*pi]
!   q      -- the external momentum, recovered via q = Q_MAX*xx(3); in [0, Q_MAX]
!   q2     -- precomputed q^2 to avoid redundant multiplications
!   denom1 -- first  denominator of the integrand: k^2 + mu^2
!   denom2 -- second denominator of the integrand: k^2 - 2kq*cos(phi) + q^2 + mu^2
!   jac    -- Jacobian of all three variable mappings combined:
!             (1/(1-t)^2) * 2*pi * Q_MAX
!             Without this factor the answer would be wrong.
!   TWOPI  -- the constant 2*pi, defined at full double precision
!   Q_MAX  -- local copy of the upper limit of q (must match the main program)
!
! STEP-BY-STEP LOGIC:
!   1. If t = 1, k would be infinite -> return 0 immediately (safe boundary).
!   2. Recover k   = t / (1 - t)
!   3. Recover phi = 2*pi * xx(2)
!   4. Recover q   = Q_MAX * xx(3)
!   5. Compute Jacobian: jac = (1/(1-t)^2) * 2*pi * Q_MAX
!   6. Evaluate denom1 and denom2
!   7. Safety check: if either denominator is essentially zero, return 0.
!   8. Return ff(1) = (k * q^2 / (denom1 * denom2)) * jac
!------------------------------------------------------------------------------
subroutine integrand_I3(ndim, xx, ncomp, ff, userdata)
  use params_mod
  implicit none

  integer, intent(in)  :: ndim          ! number of dimensions (fixed by Cuba)
  real(8), intent(in)  :: xx(ndim)      ! integration point in [0,1]^3
  integer, intent(in)  :: ncomp         ! number of integrand components (fixed by Cuba)
  real(8), intent(out) :: ff(ncomp)     ! integrand value to be filled
  integer, intent(in)  :: userdata      ! unused extra argument required by Cuba interface

  real(8), parameter :: TWOPI = 6.28318530717958647692d0
  real(8), parameter :: Q_MAX = 5.0d0   ! upper limit of q integration (must match main)

  real(8) :: t       ! mapped variable for k: t in [0, 1)
  real(8) :: k       ! integration variable: radial momentum, k in [0, +inf)
  real(8) :: phi     ! integration variable: azimuthal angle, phi in [0, 2*pi]
  real(8) :: q       ! integration variable: external momentum, q in [0, Q_MAX]
  real(8) :: q2      ! q^2, precomputed to avoid redundant multiplication
  real(8) :: denom1  ! first  denominator:  k^2 + mu^2
  real(8) :: denom2  ! second denominator:  k^2 - 2kq*cos(phi) + q^2 + mu^2
  real(8) :: jac     ! Jacobian: (1/(1-t)^2) * 2*pi * Q_MAX

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

  ! External momentum: q = Q_MAX * s
  q = Q_MAX * xx(3)

  ! Precompute q^2
  q2 = q * q

  ! Jacobian: dk/dt * dphi/dr * dq/ds = 1/(1-t)^2 * 2*pi * Q_MAX
  jac = TWOPI * Q_MAX / (1d0 - t)**2

  ! Evaluate the two denominators of the physical integrand
  denom1 = k*k + mu_val*mu_val
  denom2 = k*k - 2d0*k*q*cos(phi) + q2 + mu_val*mu_val

  ! Safety check: avoid division by zero (should not occur for mu > 0)
  if (abs(denom1) < 1d-30 .or. abs(denom2) < 1d-30) then
    ff(1) = 0d0
    return
  end if

  ! Physical integrand times Jacobian
  ff(1) = (k * q2 / (denom1 * denom2)) * jac

end subroutine integrand_I3
