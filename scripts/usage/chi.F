       program chi_b_scan
c --------------------------------------
c  Calcula chi(s,b) em funcao de b:
c
c  chi(s,b) = (1/s) * integral_0^3 q' dq' J0(b*q') A_born(s,-(q')^2)
c
c  A_born(s,t) = i * s^trajPP * 8 * diff_t
c  trajPP = 1 + epsilon - alpha' * q'^2
c  diff_t = integral 2D (cuhre) — mesmo integrando do sigma_diff.F
c
c  Parametros fisicos: identicos ao sigma_diff.F
c     sqrt_s = 7000, mg=0.389, a1=1.50, a2=2.13
c     epsilon=0.0616, alpha_prime=0.25
c
c  Integracao em q': wgauss (metodo do ger4), q' de 0 a 3
c  Varredura em b:  b de 0 a 10, passo 0.1
c
c  Saida: b   Re(chi)   Im(chi)
c --------------------------------------
      implicit none
      double precision sq_s, sqrt_s, tt, qq2
      double precision mg, a1, a2
      double precision b_val, b_step, b_max
      double precision q_min, q_max
      double precision pi
      double complex chi_val, chi_int_func
      external chi_int_func

      common/cmener/sq_s, tt
      common/ener/qq2
      common/mass/mg, a1, a2
      common/impact/b_val

c     --- interface para wgauss ---
      double complex wgauss
      double precision chi_mod
      external wgauss

c     --- parametros fisicos (identicos ao sigma_diff.F) ---
      pi        = 4.d0*datan(1.d0)
      sqrt_s    = 7000.d0
      sq_s      = sqrt_s * sqrt_s
      tt        = 0.d0
      mg        = 0.389d0
      a1        = 1.50d0
      a2        = 2.13d0

c     --- limites de integracao em q' ---
      q_min = 0.d0
      q_max = 10.d0

c     --- varredura em b ---
      b_step = 0.1d0
      b_max  = 15.d0
      b_val  = 0.d0

      write(*,'(A)')
     & '# b [GeV^-1]      Re(chi)        Im(chi)        |chi|'

      do while (b_val .le. b_max + 1.d-10)

        chi_val = wgauss(chi_int_func, q_min, q_max, 1.d-4) / sq_s

        chi_mod = cdabs(chi_val)

        write(*,'(4(2X,G16.8))')
     &    b_val, dreal(chi_val), dimag(chi_val), chi_mod

        b_val = b_val + b_step
      end do

      end program


c -----------------------------------------------------------------------
c  chi_int_func(q):
c  Integrando da integral em q' para chi:
c     f(q') = q' * J0(b*q') * A_born(s, -(q')^2)
c  A_born calculada via cuhre (mesmo integrand do sigma_diff.F)
c -----------------------------------------------------------------------
      function chi_int_func(q)
      implicit none
      double precision q, sq_s, tt, qq2, bb
      double precision mg, a1, a2, diff_t
      double precision pi, trajPP, epsilon, alpha_prime
      double complex chi_int_func, amp_born, imagi
      COMPLEX(8) :: cb(0:1), j0

c     --- cuba (cuhre) ---
      integer ndim, ncomp, nvec, last, seed, mineval, maxeval
      double precision epsrel, tt_step, userdata
      parameter (ndim     = 2)
      parameter (ncomp    = 1)
      parameter (userdata = 0)
      parameter (nvec     = 1)
      parameter (epsrel   = 1D-3)
      parameter (tt_step  = 1D-16)
      parameter (last     = 4)
      parameter (seed     = 0)
      parameter (mineval  = 50000)
      parameter (maxeval  = 10000000)
      integer nstart, nincrease, nbatch, gridno
      integer*8 spin
      character*(*) statefile
      parameter (nstart    = 1000)
      parameter (nincrease = 500)
      parameter (nbatch    = 1000)
      parameter (gridno    = 0)
      parameter (statefile = "")
      parameter (spin      = -1)
      integer key
      parameter (key = 0)
      external integrand
      double precision integral(ncomp), error(ncomp), prob(ncomp)
      integer verbose, nregions, neval, fail

      common/cmener/sq_s, tt
      common/ener/qq2
      common/mass/mg, a1, a2
      common/impact/bb

      pi           = 4.d0*datan(1.d0)
      imagi        = dcmplx(0.d0, 1.d0)
      epsilon      = 0.0616d0
      alpha_prime  = 0.25d0

c     --- q'^2 para o integrando cuhre ---
      qq2 = q * q

c     --- Bessel J0(b*q') ---
      cb(:) = 0.d0
      if (bb*q .le. 1.d-30) then
        j0 = (1.d0, 0.d0)
      else
        CALL DBSJA(bb*q, 0.d0, 1, 10, cb)
        j0 = cb(0)
      end if

c     --- diff_t via cuhre (identico ao sigma_diff.F) ---
      integral(1) = 0.d0
      call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &  epsrel, tt_step, verbose + last,
     &  mineval, maxeval, key,
     &  statefile, spin,
     &  nregions, neval, fail, integral, error, prob)
      diff_t = integral(1)

c     --- A_born = i * s^trajPP * 8 * diff_t ---
      trajPP   = 1.d0 + epsilon - alpha_prime * qq2
      amp_born = imagi * (sq_s**trajPP) * 8.d0 * diff_t

c     --- integrando: q' * J0(b*q') * A_born ---
      chi_int_func = q * j0 * amp_born

      return
      end function


c -----------------------------------------------------------------------
c  integrand: identico ao sigma_diff.F (modelo power-law)
c -----------------------------------------------------------------------
      integer function integrand(ndim, xx, ncomp, ff)
      implicit none
      integer ndim, ncomp
      double precision xx(*), ff(*)
      double precision A1MAX, A2MAX, JCB, U, V
      double precision aa1, aa2, Qt, Qt2, mm, mg, mm2
      double precision qq2, Gp_q_k, Gp_q_0, q_inf2, q_sup2
      double precision Umin, Vmin
      double precision alfaprop_sup, alfaprop_inf
      double precision lbd, lbd2, bep
      double precision Md2, Md2inf, Md2sup
      double precision a1, a2
      double precision pi, sq_s, tt
      double precision rho, rho_mg2, ratio_sup, ratio_inf, ratio_Qt
      double precision gamma2_m1
      parameter (pi = 3.14159265358979323846D0)

      common/cmener/sq_s, tt
      common/ener/qq2
      common/mass/mg, a1, a2

      mm  = mg
      aa1 = a1
      aa2 = a2
      Qt2 = qq2
      Qt  = dsqrt(Qt2)
      mm2 = mm*mm
      lbd = 0.284d0
      lbd2 = lbd*lbd
      bep  = 27.d0/(12.d0*pi)
      rho       = 4.d0
      gamma2_m1 = 1.36d0
      rho_mg2   = rho * mm2

#define x xx(1)
#define y xx(2)
#define f ff(1)

      A1MAX = sq_s
      A2MAX = 2.d0*pi
      JCB   = A1MAX*A2MAX
      U     = A1MAX*x
      V     = A2MAX*y
      Umin  = 0.d0
      Vmin  = 0.d0

      Umin = 0.d0
      if (U .ge. Umin) then
        Vmin = 0.d0
        if (V .ge. Vmin) then

          q_sup2 = (Qt2/4.d0) + U*Qt*dcos(V) + U*U
          q_inf2 = (Qt2/4.d0) - U*Qt*dcos(V) + U*U

          ratio_Qt  = dlog((Qt2   + rho_mg2)/lbd2)/dlog(rho_mg2/lbd2)
          Md2       = (mm2*mm2/(Qt2   + mm2)) * ratio_Qt**gamma2_m1

          ratio_sup = dlog((q_sup2 + rho_mg2)/lbd2)/dlog(rho_mg2/lbd2)
          Md2sup    = (mm2*mm2/(q_sup2 + mm2)) * ratio_sup**gamma2_m1

          ratio_inf = dlog((q_inf2 + rho_mg2)/lbd2)/dlog(rho_mg2/lbd2)
          Md2inf    = (mm2*mm2/(q_inf2 + mm2)) * ratio_inf**gamma2_m1

          alfaprop_sup = 1.d0/(bep*(q_sup2 + Md2sup)*
     *      dlog((q_sup2 + 4.d0*Md2sup)/lbd2))

          alfaprop_inf = 1.d0/(bep*(q_inf2 + Md2inf)*
     *      dlog((q_inf2 + 4.d0*Md2inf)/lbd2))

          Gp_q_0 = dexp(-aa1*Qt2 - aa2*Qt2*Qt2)

          Gp_q_k = dexp(-aa1*(Qt2 + 9.d0*dabs(U*U - Qt2/4.d0))
     *      -aa2*(Qt2 + 9.d0*dabs(U*U - Qt2/4.d0))*
     *           (Qt2 + 9.d0*dabs(U*U - Qt2/4.d0)))

          f = U * alfaprop_sup * alfaprop_inf *
     *      (Gp_q_0*Gp_q_0 - Gp_q_k*(2.d0*Gp_q_0 - Gp_q_k)) * JCB

        else
          f = 0.d0
        end if
      else
        f = 0.d0
      end if

      integrand = 0
      end