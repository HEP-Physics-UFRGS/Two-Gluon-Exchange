       program chi
c --------------------------------------
c      Integral tripla usando VEGAS
c      Variaveis (X,Y,Q2)
c
c   Ver pagina 174, equacao (8.46)
c
c --------------------------------------
c       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       implicit none
       double complex chi2,amp_eik,amp_eik_val
       double precision tt_step,b_min,b_max,sq_s,sqrt_s,b_val,tt
       double precision pi
       external chi2
       common/cmener/sq_s,tt
       sqrt_s=7000.d0
       sq_s=sqrt_s*sqrt_s
       tt_step=1d-3
       b_min=0.d0
       b_max=10.d0
       b_val=0.d0
       tt=0.d0
       pi=4.d0*datan(1.d0)
       do

       	do
       	amp_eik_val=chi2(b_val)
       	amp_eik=amp_eik+amp_eik_val
       	write(*,*)b_val,amp_eik
       	b_val=b_val+1.d0
       	if (b_val.le.10.d0) then
       	continue
       	else 
       	exit
       	end if
       	end do
       write(*,*)tt,amp_eik,"----------------------------------"
       tt=tt+tt_step

       amp_eik=0.d0
       b_val=0.d0
       if (tt.le.0.1d0) then
       continue
       else 
       exit
       end if
       end do

       end program
  
       function chi2(b_val)
       implicit none
       double precision tt_step,q_min,q_max,sq_s,b_val,bb,tt
       double complex chi2,chi2_int,wgauss,chi_val
       COMPLEX(8) :: cd(0:1),j0
       double complex imagi
       external chi2_int
       common/cmener/sq_s,tt
       common/impact/bb
       imagi=dcmplx(0.d0,1.d0)
       bb=b_val
       tt_step=1d-4
       q_min=0.d0
       q_max=0.2d0
       cd(:)=0.d0
!       
       if (bb*dsqrt(tt).le.1.d-30) then
       j0=(1.d0,0.d0)
       else
       CALL DBSJA(bb*dsqrt(tt),0.d0,1,10,cd)
       j0=cd(0)
       continue
       end if 

       chi_val=wgauss(chi2_int,q_min,q_max,tt_step)/sq_s
       chi2=bb*j0*(1.d0-cdexp(imagi*chi_val))

       return
       end 
c
c       
       function chi2_int(q)
      double precision pi, q, sq_s, tt, qq2, be, bep, mg,
     &                 trajPP, deltaPP, alpha_prime, a1, a2, bb, diff_t

       double complex imagi,amp_born,chi2_int
       COMPLEX(8) :: cb(0:1),j0
       integer ndim, ncomp, nvec, last, seed, mineval, maxeval
	double precision epsrel, tt_step, userdata

       
	parameter (ndim = 2)
	parameter (ncomp = 1)
	parameter (userdata = 0)
	parameter (nvec = 1)
	parameter (epsrel = 1D-3)
	parameter (tt_step = 1D-16)
	parameter (last = 4)
	parameter (seed = 0)
	parameter (mineval = 50000)
	parameter (maxeval = 10000000)
c
	integer nstart, nincrease, nbatch, gridno
	integer*8 spin
	character*(*) statefile
	parameter (nstart = 1000)
	parameter (nincrease = 500)
	parameter (nbatch = 1000)
	parameter (gridno = 0)
	parameter (statefile = "")
	parameter (spin = -1)
c
	integer key
	parameter (key = 0)

	external integrand

	double precision integral(ncomp), error(ncomp), prob(ncomp)
	integer verbose, nregions, neval, fail
       common/cmener/sq_s,tt
       common/impact/bb
       common/ener/qq2
       common/mass/mg,a1,a2
       pi=4.d0*datan(1.d0)
       imagi=dcmplx(0.d0,1.d0)
       qq2=q*q
       be=27.d0/(48.d0*pi*pi)   !n_f=3
       bep=27.d0/(12.d0*pi)   !bep=4*pi*be
c
c
       deltaPP=0.086557d0   ! Marcela
       alpha_prime=0.25d0
       mg=0.44137d0   !0.37965d0
       a1=1.6711d0   !1.5d0
       a2=2.0001d0   !2.7d0
c
       cb(:)=0.d0
!       
       if (bb*q.le.1.d-30) then
       j0=(1.d0,0.d0)
       else
       CALL DBSJA(bb*q,0.d0,1,10,cb)
       j0=cb(0)
       continue
       end if     

       integral(1)=0.d0
c
c
       call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, tt_step, verbose + last,
     &    mineval, maxeval, key,
     &    statefile, spin,
     &    nregions, neval, fail, integral, error, prob)
       diff_t=integral(1)
       trajPP=1.d0+deltaPP-alpha_prime*qq2
       amp_born=imagi*(sq_s**trajPP)*8.d0*diff_t
       chi2_int=q*j0*amp_born
      return
      END function
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
	parameter (pi = 3.14159265358979323846D0)
       
	common/cmener/sq_s,tt
	common/ener/qq2
        common/mass/mg,a1,a2
        mm=mg   !FCN=25897.83
       aa1=a1
       aa2=a2
       Qt2=qq2
       Qt=dsqrt(Qt2)
       mm2=mm*mm
       lbd=0.284d0
       lbd2=lbd*lbd
       be=27.d0/(48.d0*pi*pi)   !n_f=3
       bep=27.d0/(12.d0*pi)   !bep=4*pi*be
       m_p=0.938272029d0
       m_p2=m_p*m_p

#define x xx(1)
#define y xx(2)
!#define z xx(3)
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
       Md2=(mm2*mm2/(Qt2+mm2))*((dlog(4.d0*mm2/lbd2)/
     *   dlog((Qt2+4.d0*mm2)/lbd2))**(-1.36d0))    
c
       Md2sup=(mm2*mm2/(q_sup2+mm2))*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_sup2+4.d0*mm2)/lbd2))**(-1.36d0))
c
       Md2inf=(mm2*mm2/(q_inf2+mm2))*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_inf2+4.d0*mm2)/lbd2))**(-1.36d0))
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