      program chi_b_scan
c --------------------------------------
c  Calcula chi(s,b) em funcao de b:
c
c  chi(s,b) = (1/s) * integral_0^q_max q' dq' J0(b*q') A_born(s,-(q')^2)
c
c  A_born(s,t) = i * s^trajPP * 8 * diff_t
c  trajPP = 1 + epsilon - alpha' * q'^2
c  diff_t = integral 2D (cuhre) — mesmo integrando do sigma_diff.F
c
c  Parametros fisicos: identicos ao sigma_diff.F
c     sqrt_s = 7000, mg=0.389, a1=1.50, a2=2.13
c     epsilon=0.0616, alpha_prime=0.25
c
c  Integracao em q': metodo de Filon composto
c    A integral contem o kernel oscilatório J0(b*q').
c    Filon separa a parte lenta g(q')=q'*A_born do kernel e usa
c    coeficientes analiticos para cos/sin, evitando cancelamentos
c    numericos que degradam Simpson para b grande.
c    J0(bq) = Re[ exp(i*b*q) ] (representacao complexa usada internamente).
c    N deve ser par (2M subintervalos, passo h, theta=b*h).
c
c  Varredura em b:  b de 0 a 15, passo 0.1
c
c  Saida: b   Re(chi)   Im(chi)   |chi|
c --------------------------------------
      implicit none
      double precision sq_s, sqrt_s, tt, qq2
      double precision mg, a1, a2
      double precision b_val, b_step, b_max
      double precision q_min, q_max
      double precision pi
      double complex chi_val
      double precision chi_mod

      common/cmener/sq_s, tt
      common/ener/qq2
      common/mass/mg, a1, a2
      common/impact/b_val

c     --- parametros de Filon ---
c     N_filon: numero de subintervalos (deve ser par, 2M pares de passos)
      integer N_filon
      parameter (N_filon = 1000)

c     --- parametros fisicos ---
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

        call filon_chi(q_min, q_max, N_filon, chi_val)
        chi_val = chi_val / sq_s

        chi_mod = cdabs(chi_val)

        write(*,'(4(2X,G16.8))')
     &    b_val, dreal(chi_val), dimag(chi_val), chi_mod

        b_val = b_val + b_step
      end do

      end program


c -----------------------------------------------------------------------
c  filon_chi: integra chi(b) = integral_q_min^q_max g(q)*J0(b*q) dq
c
c  onde  g(q) = q * A_born(s, -q^2)   (parte suave, complexa)
c
c  Metodo de Filon composto (Filon 1928):
c    Decompoe J0(bq) = Re[e^{i*b*q}] e aplica quadratura de Filon
c    separadamente para as partes cossenoidal e seno:
c       I_cos = integral g(q) cos(b*q) dq
c       I_sin = integral g(q) sin(b*q) dq
c    de modo que  integral g(q) J0(b*q) dq = I_cos  (parte real de J0)
c    Para b*h pequeno os coeficientes alpha,beta,gamma convergem para
c    os de Simpson, garantindo continuidade numerica.
c
c  Formulacao composta sobre 2M subintervalos de passo h (N = 2M, par):
c    theta = b*h
c    alpha = (1/theta + sin(2*theta)/(2*theta^2) - 2*sin^2(theta)/theta^3)
c    beta  = 2*( (1+cos^2(theta))/theta^2 - sin(2*theta)/theta^3 )
c    gamma = 4*( sin(theta)/theta^3 - cos(theta)/theta^2 )
c
c  Caso b=0 ou theta<1e-6: usa limite analitico (coef. de Simpson).
c
c  N deve ser par.
c -----------------------------------------------------------------------
      subroutine filon_chi(q_min, q_max, N, result)
      implicit none
      double precision q_min, q_max
      integer N
      double complex result

      double precision h, theta, q_i
      double precision alpha_f, beta_f, gamma_f
      double precision s_th, c_th, s2th, c2th
      double complex g0, g_end, g_last
      double complex S_odd, S_even, g_i
      double complex amp_born_func
      double complex I_cos, I_sin
      integer i, M
      external amp_born_func

      double precision bb
      common/impact/bb

      if (mod(N,2) .ne. 0) then
        write(*,*) 'AVISO: N_filon deve ser par. Incrementando para',
     &             N+1
        N = N + 1
      end if
      M = N / 2

      h     = (q_max - q_min) / dble(N)
      theta = bb * h

c     --- coeficientes de Filon ---
      if (dabs(theta) .lt. 1.d-6) then
c       limite theta->0: recupera coeficientes de Simpson
        alpha_f = 0.d0
        beta_f  = 2.d0/3.d0
        gamma_f = 4.d0/3.d0
      else
        s_th  = dsin(theta)
        c_th  = dcos(theta)
        s2th  = dsin(2.d0*theta)
        c2th  = dcos(2.d0*theta)
        alpha_f = ( 1.d0/theta
     &            + s2th/(2.d0*theta**2)
     &            - 2.d0*s_th**2/theta**3 )
        beta_f  = 2.d0*( (1.d0 + c_th**2)/theta**2
     &                  - s2th/theta**3 )
        gamma_f = 4.d0*( s_th/theta**3 - c_th/theta**2 )
      end if

c     --- acumula somas separadas para cos e sin ---
c     g(q) = amp_born_func(q)  — retorna q * A_born(s,-q^2), sem J0
      g0    = amp_born_func(q_min)
      g_end = amp_born_func(q_max)
      g_last = amp_born_func(q_max - h)   ! ponto q_{N-1}

c     S_odd  = soma nos indices impares  1,3,5,...,N-1  (coef. gamma)
c     S_even = soma nos indices pares    2,4,6,...,N-2  (coef. beta)
      S_odd  = dcmplx(0.d0, 0.d0)
      S_even = dcmplx(0.d0, 0.d0)

      do i = 1, N-1
        q_i = q_min + dble(i)*h
        g_i = amp_born_func(q_i)
        if (mod(i,2) .eq. 1) then
          S_odd  = S_odd  + g_i * dcmplx( dcos(bb*q_i),
     &                                   -dsin(bb*q_i) )
        else
          S_even = S_even + g_i * dcmplx( dcos(bb*q_i),
     &                                   -dsin(bb*q_i) )
        end if
      end do

c     --- termos dos extremos com fase ---
      I_cos = h * (
     &   alpha_f * ( g_end*dcmplx( dcos(bb*q_max),
     &                             -dsin(bb*q_max) )
     &             - g0  *dcmplx( dcos(bb*q_min),
     &                            -dsin(bb*q_min) ) )
     & + beta_f  * ( g_end*dcmplx( dcos(bb*q_max),
     &                             -dsin(bb*q_max) )
     &             + g0  *dcmplx( dcos(bb*q_min),
     &                            -dsin(bb*q_min) ) )
     & + gamma_f * S_odd
     & + beta_f  * S_even )

c     O metodo de Filon classico calcula diretamente a integral complexa
c     integral g(q) * e^{-i*b*q} dq  e toma a parte real (= integral com J0)
c     Aqui acumulamos tudo em forma complexa e devolvemos a parte real
c     como I_cos; a parte imaginaria (com sin) e descartada pois J0 = Re[e^{ibq}]

      result = dreal(I_cos)

      return
      end subroutine


c -----------------------------------------------------------------------
c  amp_born_func(q):
c  Retorna a parte suave do integrando: g(q) = q * A_born(s, -q^2)
c  SEM o fator J0 — este e tratado analiticamente pelo metodo de Filon.
c  A_born calculada via cuhre (identico ao sigma_diff.F)
c -----------------------------------------------------------------------
      function amp_born_func(q)
      implicit none
      double precision q, sq_s, tt, qq2
      double precision mg, a1, a2, diff_t
      double precision pi, trajPP, epsilon, alpha_prime
      double complex amp_born_func, amp_born, imagi

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

      pi           = 4.d0*datan(1.d0)
      imagi        = dcmplx(0.d0, 1.d0)
      epsilon      = 0.0616d0
      alpha_prime  = 0.25d0

c     --- q^2 para o integrando cuhre ---
      qq2 = q * q

c     --- diff_t via cuhre ---
      integral(1) = 0.d0
      verbose = 0
      call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &  epsrel, tt_step, verbose + last,
     &  mineval, maxeval, key,
     &  statefile, spin,
     &  nregions, neval, fail, integral, error, prob)
      diff_t = integral(1)

c     --- A_born = i * s^trajPP * 8 * diff_t ---
      trajPP   = 1.d0 + epsilon - alpha_prime * qq2
      amp_born = imagi * (sq_s**trajPP) * 8.d0 * diff_t

c     --- parte suave: g(q) = q * A_born (sem J0) ---
      amp_born_func = q * amp_born

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