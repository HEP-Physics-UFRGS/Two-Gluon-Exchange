	program diff_t_scan
c --------------------------------------
c  Avalia diff_t e A_born em funcao de q
c  q varia de 0 a 0.2, com passo q_step
c
c  diff_t  = integral(cuhre) — integracao 2D (integrand)
c  trajPP  = 1 + epsilon - alpha' * q^2
c  A_born  = i * s^trajPP * 8 * diff_t
c
c  Modelo de massa dinamica: m2_pl (power-law)
c  Md2 = (mg^4/(q2+mg^2)) * (log((q2+rho*mg^2)/lbd^2)/log(rho*mg^2/lbd^2))^1.36
c --------------------------------------
       implicit none
       double precision sq_s, sqrt_s, tt, qq2
       double precision mg, a1, a2
       double precision q, q_step, diff_t
       double precision pi, trajPP, epsilon, alpha_prime
       double complex amp_born, imagi
       common/cmener/sq_s,tt
       common/ener/qq2
       common/mass/mg,a1,a2
c
       integer ndim, ncomp, nvec, last, seed, mineval, maxeval
       double precision epsrel, tt_step, userdata
       parameter (ndim    = 2)
       parameter (ncomp   = 1)
       parameter (userdata= 0)
       parameter (nvec    = 1)
       parameter (epsrel  = 1D-3)
       parameter (tt_step = 1D-16)
       parameter (last    = 4)
       parameter (seed    = 0)
       parameter (mineval = 50000)
       parameter (maxeval = 10000000)
c
       integer nstart, nincrease, nbatch, gridno
       integer*8 spin
       character*(*) statefile
       parameter (nstart   = 1000)
       parameter (nincrease= 500)
       parameter (nbatch   = 1000)
       parameter (gridno   = 0)
       parameter (statefile= "")
       parameter (spin     = -1)
c
       integer key
       parameter (key = 0)
c
       external integrand
       double precision integral(ncomp), error(ncomp), prob(ncomp)
       integer verbose, nregions, neval, fail
c
c      --- parametros fisicos ---
       pi          = 4.d0*datan(1.d0)
       imagi       = dcmplx(0.d0,1.d0)
       sqrt_s      = 7000.d0
       sq_s        = sqrt_s*sqrt_s
       tt          = 0.d0
c      Parametros ajustados (compativel com Python, modelo pl, atlas)
       mg          = 0.389d0
       a1          = 1.50d0
       a2          = 2.13d0
       epsilon     = 0.0616d0
       alpha_prime = 0.25d0
c
c      --- varredura em q2 (Python itera em q2, nao em q) ---
c          Python: q2 de 0.01 ate 0.2001, passo 0.001
c          Fortran original iterava em q com passo 0.01
c          Agora corrigido para iterar em q2 como o Python
       q_step  = 0.001d0
       q       = 0.1d0      ! q = sqrt(q2_inicial) = sqrt(0.01)
c
       write(*,'(A)')
     & '========================='
       write(*,'(A)')
     & '   q2 [GeV2]    diff_t         error'//
     & '        Re(A_born)      Im(A_born)'
       write(*,'(A)')
     & '========================='
c
c      Itera em q2 de 0.01 a 0.2001, passo 0.001 (igual ao Python)
       qq2 = 0.01d0
       do while (qq2 .le. 0.2001d0)
c
         integral(1) = 0.d0
c
         call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &     epsrel, tt_step, verbose + last,
     &     mineval, maxeval, key,
     &     statefile, spin,
     &     nregions, neval, fail, integral, error, prob)
c
         diff_t  = integral(1)
c
c        A_born = i * s^trajPP * 8 * diff_t
c        trajPP = 1 + epsilon + alpha_prime * t,  t = -q2
         trajPP   = 1.d0 + epsilon - alpha_prime * qq2
         amp_born = imagi * (sq_s**trajPP) * 8.d0 * diff_t
c
         write(*,'(5(2X,E14.6))')
     &     qq2, diff_t, error(1),
     &     dreal(amp_born), dimag(amp_born)
c
         qq2 = qq2 + q_step
       end do
c
       write(*,'(A)')
     & '========================='
c
       end program
c
c
************************************************************************

	integer function integrand(ndim, xx, ncomp, ff)
	implicit none
	integer ndim, ncomp
	double precision xx(*), ff(*)
	double precision A1MAX,A2MAX,JCB,U,V
       double precision aa1,aa2,Qt,Qt2,mm,mg,mm2
       double precision qq2,Gp_q_k,Gp_q_0,q_inf2,q_sup2
       double precision Umin,Vmin,m_p,m_p2
       double precision alfaprop_sup,alfaprop_inf
       double precision lbd,lbd2, be,bep
       double precision Md2,Md2inf,Md2sup
       double precision a1,a2
       double precision pi,sq_s,tt
       double precision rho, rho_mg2, ratio_sup, ratio_inf, ratio_Qt
       double precision gamma2_m1
	parameter (pi = 3.14159265358979323846D0)
       
	common/cmener/sq_s,tt
	common/ener/qq2
        common/mass/mg,a1,a2
        mm=mg
       aa1=a1
       aa2=a2
       Qt2=qq2
       Qt=dsqrt(Qt2)
       mm2=mm*mm
       lbd=0.284d0
       lbd2=lbd*lbd
c      b_0 = (33-6)/(12*pi) = 27/(12*pi)
       bep=27.d0/(12.d0*pi)
c      Parametros do modelo power-law (pl)
       rho       = 4.d0
       gamma2_m1 = 1.36d0    ! gamma_2 - 1 = 2.36 - 1
       rho_mg2   = rho * mm2

#define x xx(1)
#define y xx(2)
#define f ff(1)


	A1MAX=sq_s
	A2MAX=2.d0*pi
	JCB=A1MAX*A2MAX
	U=A1MAX*x
	V=A2MAX*y
	Umin=0.d0
	Vmin=0.d0
c	
c	
       Umin=0.d0
       if (U.ge.Umin) then
c
       Vmin=0.d0
       if (V.ge.Vmin) then   
c
       q_sup2=(Qt2/4.d0)+U*Qt*dcos(V)+U*U
c
       q_inf2=(Qt2/4.d0)-U*Qt*dcos(V)+U*U
c
c      Modelo m2_pl: Md2 = (mg^4/(q2+mg^2)) * ratio^(gamma2-1)
c      ratio = log((q2 + rho*mg^2)/lbd^2) / log(rho*mg^2/lbd^2)
c
       ratio_Qt  = dlog((Qt2   + rho_mg2)/lbd2) / dlog(rho_mg2/lbd2)
       Md2       = (mm2*mm2/(Qt2   + mm2)) * ratio_Qt**gamma2_m1
c
       ratio_sup = dlog((q_sup2 + rho_mg2)/lbd2) / dlog(rho_mg2/lbd2)
       Md2sup    = (mm2*mm2/(q_sup2 + mm2)) * ratio_sup**gamma2_m1
c
       ratio_inf = dlog((q_inf2 + rho_mg2)/lbd2) / dlog(rho_mg2/lbd2)
       Md2inf    = (mm2*mm2/(q_inf2 + mm2)) * ratio_inf**gamma2_m1
c
       alfaprop_sup=1.d0/(bep*(q_sup2+Md2sup)*
     *  dlog((q_sup2+4.d0*Md2sup)/lbd2))
c
       alfaprop_inf=1.d0/(bep*(q_inf2+Md2inf)*
     *  dlog((q_inf2+4.d0*Md2inf)/lbd2))
c
       Gp_q_0=dexp(-aa1*Qt2-aa2*Qt2*Qt2)
c
       Gp_q_k=dexp(-aa1*(Qt2+9.d0*dabs(U*U-Qt2/4.d0))
     *   -aa2*(Qt2+9.d0*dabs(U*U-Qt2/4.d0))*
     *   (Qt2+9.d0*dabs(U*U-Qt2/4.d0)))
c
       f=U*alfaprop_sup*alfaprop_inf*
     *   (Gp_q_0*Gp_q_0-Gp_q_k*(2.d0*Gp_q_0-Gp_q_k))*JCB
c
       else
       f=0.d0
       endif
       else
       f=0.d0
       endif
c
c
	integrand = 0
	end
