!==============================================================================
! I_4.f90
!
! WHAT THIS PROGRAM DOES:
!   Numerically computes a 3D integral (three nested integrals) that appears
!   in physics calculations. The integral has THREE variables:
!     - k   : a radial momentum, ranging from 0 to K_CUT (finite cutoff = 10)
!     - phi : an angle, ranging from 0 to 2*pi (a full circle)
!     - q   : an external momentum, ranging from 0 to Q_MAX (= 5)
!
!   The mathematical formula being evaluated is:
!
!     I4 = INT_0^5 dq  INT_0^{2pi} dphi  INT_0^{10} dk
!               k * q^2 / [ (k^2 + mu^2) * (k^2 - 2kq*cos(phi) + q^2 + mu^2) ]
!
! HOW IT DIFFERS FROM THE OTHER FILES:
!   - I_1.f90: 2D integral (k and phi),       k goes to infinity
!   - I_2.f90: 2D integral (k and phi),       k has a finite cutoff K_CUT
!   - I_3.f90: 3D integral (k, phi, and q),   k goes to infinity
!   - I_4.f90: 3D integral (k, phi, and q),   ALL limits are finite (simplest case)
!
! HOW IT HANDLES THE INTEGRATION RANGES:
!   Because ALL three variables have finite limits, NO special substitution
!   trick is needed anywhere. All three mappings are simple linear rescalings:
!
!       k   = K_CUT * s,    s in [0, 1]    (stretches [0,1] to [0, K_CUT])
!       phi = 2*pi  * r,    r in [0, 1]    (stretches [0,1] to [0, 2*pi])
!       q   = Q_MAX * t,    t in [0, 1]    (stretches [0,1] to [0, Q_MAX])
!
!   The combined Jacobian for all three linear mappings is:
!       J = K_CUT * 2*pi * Q_MAX
!
!   There is no boundary singularity to handle and no special edge cases.
!   This is the most straightforward version of the integrand subroutine.
!
! EXTERNAL LIBRARY USED:
!   Cuba (CUHRE) -- a professional library for multidimensional numerical
!   integration. Download: https://feynarts.de/cuba/
!
! HOW TO COMPILE:
!   gfortran -O2 -o integral_I4 integral_I4.f90 -lcuba -lm
!
! HOW TO RUN:
!   Edit the value of MU in the "USER INPUT" section of the main program
!   below, then recompile and run:  ./integral_I4
!==============================================================================


!------------------------------------------------------------------------------
! MODULE: params_mod
!
! WHAT IT IS:
!   A shared storage box that holds two numbers the rest of the program needs.
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
!   kcut_val -- the upper limit of integration for k. Set to 10 in this program.
!               Stored here so the integrand can use it for the linear mapping
!               and for the Jacobian calculation.
!
!   Both are real(8): 64-bit double-precision decimal numbers (~15-16 significant
!   digits). The "save" attribute ensures values persist between subroutine calls.
!
! NOTE: Q_MAX is NOT stored here. It is defined as a local constant directly
!   inside the integrand subroutine (it must match the value in the main program).
!------------------------------------------------------------------------------
module params_mod
  implicit none
  real(8), save :: mu_val    ! infrared regulator / gluon mass parameter
  real(8), save :: kcut_val  ! upper integration limit passed to the integrand
end module params_mod


!------------------------------------------------------------------------------
! PROGRAM: integral_I4
!
! WHAT IT DOES:
!   This is the main program -- execution starts here. It:
!     1. Stores the physical parameters (MU, K_CUT) into the shared module.
!     2. Calls the Cuba CUHRE integrator to numerically compute I4.
!     3. Prints a formatted results table to the screen.
!------------------------------------------------------------------------------
program integral_I4
  use params_mod
  implicit none

  !----------------------------------------------------------------------------
  ! USER INPUT -- edit these values before compiling
  !
  !   MU    : the infrared regulator. Must be greater than zero.
  !   Q_MAX : the upper limit of the q integration.
  !   K_CUT : the upper limit of the k integration.
  !
  ! IMPORTANT: if you change Q_MAX here, you must also change the Q_MAX
  ! constant inside the integrand_I4 subroutine below to match.
  !----------------------------------------------------------------------------
  real(8), parameter :: MU    = 1.0d0    ! infrared regulator (must be > 0)
  real(8), parameter :: Q_MAX = 5.0d0    ! upper limit of q integration

  ! Upper cutoff for the k integration.
  real(8), parameter :: K_CUT = 10.0d0

  !----------------------------------------------------------------------------
  ! CUBA CUHRE SETTINGS
  !
  ! These control how the numerical integrator behaves. You usually do not
  ! need to change these unless you want different precision or performance.
  !
  !   NDIM    : number of integration dimensions. 3 here (k, phi, AND q).
  !   NCOMP   : number of integrands to compute. 1 here (just I4).
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
  integer, parameter :: NCOMP   = 1          ! number of integrands (we have just one: I4)
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
  !   I4          : the final answer (equals integral(1) directly, since all
  !                 limits are finite and no tail correction is needed)
  !----------------------------------------------------------------------------
  integer :: nregions           ! number of subregions used during integration
  integer :: neval              ! actual number of integrand evaluations performed
  integer :: fail               ! convergence flag: 0 = converged, 1 = max evals reached

  real(8) :: integral(NCOMP)   ! estimated value of the integral
  real(8) :: error(NCOMP)      ! estimated absolute error on integral
  real(8) :: prob(NCOMP)       ! chi-square probability (reliability of error estimate)
  real(8) :: I4                 ! final result

  ! PI constant
  real(8), parameter :: PI = 3.14159265358979323846d0

  ! Declaration of the external integrand subroutine (defined below)
  external :: integrand_I4

  !----------------------------------------------------------------------------
  ! Step 1: Store MU and K_CUT into the shared module so integrand_I4 can use them
  !----------------------------------------------------------------------------
  call set_params(MU, K_CUT)

  !----------------------------------------------------------------------------
  ! Step 2: Call Cuba CUHRE to integrate over the unit cube [0,1]^3.
  !
  ! Cuba will call integrand_I4 many times, each time at a different point
  ! (xx(1), xx(2), xx(3)) in [0,1]^3. Inside the integrand, those points are
  ! mapped with simple linear rescalings:
  !   xx(1) = s in [0,1]  ->  k   = K_CUT * s    covers k   in [0, K_CUT]
  !   xx(2) = r in [0,1]  ->  phi = 2*pi * r       covers phi in [0, 2*pi]
  !   xx(3) = t in [0,1]  ->  q   = Q_MAX * t      covers q   in [0, Q_MAX]
  !----------------------------------------------------------------------------
  call Cuhre(NDIM, NCOMP, integrand_I4, 0,  &  ! dimensions, components, integrand, userdata
             NVEC,                            &  ! points per batch
             EPSREL, EPSABS,                 &  ! error targets
             FLAGS,                           &  ! verbosity
             MINEVAL, MAXEVAL,               &  ! evaluation bounds
             KEY,                             &  ! cubature rule selector
             STATEFILE, SPIN,                &  ! state file, parallelisation
             nregions, neval, fail,          &  ! output: regions, evals, status
             integral, error, prob)             ! output: result, error, chi2 prob

  ! Final result (no tail correction needed: all limits finite)
  I4 = integral(1)

  !----------------------------------------------------------------------------
  ! Step 3: Print results to the screen
  !----------------------------------------------------------------------------
  write(*,*)
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a)') '  |            Cuba CUHRE  --  Result I4            |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f8.4,a,f8.4,a)') &
       '  |   mu = ', MU, '   q in [0, ', Q_MAX, ']              |'
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,'(a,f18.10,a)')  '  |   Result        ', I4,          '           |'
  write(*,'(a,es14.4,a)')  '  |   Est. error    ', error(1),    '               |'
  write(*,'(a,I42,a)')     '  |   Evaluations   ', neval,        '               |'
  write(*,'(a,I42,a)')     '  |   Subregions    ', nregions,     '               |'
  if (fail == 0) then
    write(*,'(a)') '  |   Status        CONVERGED                       |'
  else
    write(*,'(a)') '  |   Status        WARNING: max evals reached      |'
  end if
  write(*,'(a)') '  +-------------------------------------------------+'
  write(*,*)

end program integral_I4


!------------------------------------------------------------------------------
! SUBROUTINE: set_params
!
! WHAT IT DOES:
!   A simple one-time setup helper. It takes mu and kcut from the main program
!   and writes them into the shared module params_mod, making them available
!   to integrand_I4 later.
!
! ARGUMENTS:
!   mu   [input, real(8)] -- the infrared regulator value to be stored.
!                            "intent(in)" means this subroutine only reads it,
!                            never modifies it.
!   kcut [input, real(8)] -- the upper k integration limit to be stored.
!                            "intent(in)" means this subroutine only reads it,
!                            never modifies it.
!
! WHAT HAPPENS INSIDE:
!   mu_val   = mu     <- writes mu   into the shared module
!   kcut_val = kcut   <- writes kcut into the shared module
!------------------------------------------------------------------------------
subroutine set_params(mu, kcut)
  use params_mod
  implicit none
  real(8), intent(in) :: mu    ! infrared regulator
  real(8), intent(in) :: kcut  ! upper cutoff for numerical integration
  mu_val   = mu
  kcut_val = kcut
end subroutine set_params


!------------------------------------------------------------------------------
! SUBROUTINE: integrand_I4
!
! WHAT IT DOES:
!   This is the heart of the calculation. Cuba calls this subroutine thousands
!   (or millions) of times. Each call provides a point (xx(1), xx(2), xx(3))
!   in the unit cube [0,1]^3, and this subroutine must return the value of the
!   integrand at that point via ff(1).
!
! KEY FEATURE -- FULLY FINITE, ALL LINEAR MAPPINGS:
!   Because all three integration variables have finite upper limits, every
!   mapping here is a plain linear rescaling. There is no infinity-handling
!   substitution (unlike I_1.f90 and I_3.f90), no boundary singularity at
!   t = 1, and no edge case to guard against. The Jacobian is simply the
!   constant K_CUT * 2*pi * Q_MAX.
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
!   k      -- the actual momentum, recovered via k = K_CUT * xx(1); in [0, K_CUT]
!   phi    -- the azimuthal angle, recovered via phi = 2*pi * xx(2); in [0, 2*pi]
!   q      -- the external momentum, recovered via q = Q_MAX * xx(3); in [0, Q_MAX]
!   q2     -- precomputed q^2 to avoid redundant multiplications
!   denom1 -- first  denominator of the integrand: k^2 + mu^2
!   denom2 -- second denominator of the integrand: k^2 - 2kq*cos(phi) + q^2 + mu^2
!   jac    -- Jacobian of all three linear mappings: K_CUT * 2*pi * Q_MAX
!             This corrects for the rescaling of all three variables.
!             Without it the answer would be wrong.
!   TWOPI  -- the constant 2*pi, defined at full double precision
!   Q_MAX  -- local copy of the upper q limit (must match the main program)
!
! STEP-BY-STEP LOGIC:
!   1. Recover k   = K_CUT * xx(1)   (linear map from [0,1] to [0, K_CUT])
!   2. Recover phi = 2*pi  * xx(2)   (linear map from [0,1] to [0, 2*pi])
!   3. Recover q   = Q_MAX * xx(3)   (linear map from [0,1] to [0, Q_MAX])
!   4. Compute Jacobian: jac = K_CUT * 2*pi * Q_MAX
!   5. Evaluate denom1 and denom2
!   6. Safety check: if either denominator is essentially zero, return 0.
!   7. Return ff(1) = (k * q^2 / (denom1 * denom2)) * jac
!------------------------------------------------------------------------------
subroutine integrand_I4(ndim, xx, ncomp, ff, userdata)
  use params_mod
  implicit none

  integer, intent(in)  :: ndim          ! number of dimensions (fixed by Cuba)
  real(8), intent(in)  :: xx(ndim)      ! integration point in [0,1]^3
  integer, intent(in)  :: ncomp         ! number of integrand components (fixed by Cuba)
  real(8), intent(out) :: ff(ncomp)     ! integrand value to be filled
  integer, intent(in)  :: userdata      ! unused extra argument required by Cuba interface

  real(8), parameter :: TWOPI = 6.28318530717958647692d0
  real(8), parameter :: Q_MAX = 5.0d0   ! upper limit of q integration (must match main)

  real(8) :: k       ! integration variable: radial momentum, k in [0, K_CUT]
  real(8) :: phi     ! integration variable: azimuthal angle, phi in [0, 2*pi]
  real(8) :: q       ! integration variable: external momentum, q in [0, Q_MAX]
  real(8) :: q2      ! q^2, precomputed to avoid redundant multiplication
  real(8) :: denom1  ! first  denominator:  k^2 + mu^2
  real(8) :: denom2  ! second denominator:  k^2 - 2kq*cos(phi) + q^2 + mu^2
  real(8) :: jac     ! Jacobian: K_CUT * 2*pi * Q_MAX

  ! Linear mappings
  k   = kcut_val * xx(1)
  phi = TWOPI    * xx(2)
  q   = Q_MAX    * xx(3)

  ! Precompute q^2
  q2 = q * q

  ! Jacobian of all three mappings combined
  jac = kcut_val * TWOPI * Q_MAX

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

end subroutine integrand_I4