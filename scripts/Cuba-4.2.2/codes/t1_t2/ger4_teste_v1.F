       program integra3d
c --------------------------------------
c --------------------------------------
c       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       implicit none
       double precision pi,resultado
       double precision A1MAX,A2MAX,A3MAX,A1MAXt,avgi2,ACC
       double precision sig_tot53,sig_tot70,sig_tot80,sig_tot13
       double precision dif53,dif70,dif80,dif13
       double precision ss,ss53,ss70,ss80,ss13,ssroot0,deltass
       double precision integra,integr53,integr70,integr80,integr13
       double precision deltaq,qq0,qq,qq2,alfa0,alfa,g2,be,bep
       double precision mm,mm0,mm2,lbd,lbd2,Md2,trajPP,deltaPP,alfalin
       double precision aa0,aaz0,sig_tot,avgi
       double complex imagi,amplitude
       integer nptq,npts,out1,out2,out3,out4,out5,ieqqe,iesse
       integer NDIM2,NCALL1,ITMX1,NPRN,IGRAPH
       integer ndim, ncomp, nvec, last, seed, mineval, maxeval
	double precision epsrel, epsabs, userdata
	parameter (ndim = 2)
	parameter (ncomp = 1)
	parameter (userdata = 0)
	parameter (nvec = 1)
	parameter (epsrel = 1D-6)
	parameter (epsabs = 1D-16)
	parameter (last = 4)
	parameter (seed = 0)
	parameter (mineval = 50000)
	parameter (maxeval = 3000000) !1000000000
c
	integer nstart, nincrease, nbatch, gridno
	integer*8 spin
	character*(*) statefile	
	parameter (nstart = 1000)
	parameter (nincrease = 500)
	parameter (nbatch = 1000)
	parameter (gridno = 1)
	parameter (statefile = "")
	parameter (spin = -1)
c
        integer nnew, nmin
	double precision flatness
	parameter (nnew = 1000)
	parameter (nmin = 2)
	parameter (flatness = 100D0)

	integer key1, key2, key3, maxpass
	double precision border, maxchisq, mindeviation
	integer ngiven, ldxgiven, nextra
	parameter (key1 = 47)
	parameter (key2 = 1)
	parameter (key3 = 1)
	parameter (maxpass = 5)
	parameter (border = 0D0)
	parameter (maxchisq = 10D0)
	parameter (mindeviation = .25D0)
	parameter (ngiven = 0)
	parameter (ldxgiven = ndim)
	parameter (nextra = 0)
c	
	integer key
	parameter (key = 0)

	external integr,integr70

	double precision integral(ncomp), error(ncomp), prob(ncomp)
	integer verbose, nregions, neval, fail
	character*16 env
	integer c
	
       common/FER/avgi
       common/dime/NDIM2
       common/ener/qq2
       common/mass/mm0,aa0,aaz0
       pi=4.d0*datan(1.d0)
       imagi=dcmplx(0.d0,1.d0)
c
c
       deltaPP=0.0892d0   ! Marcela
       mm0=0.38d0   !0.37965d0
       aa0=1.491d0   !1.5d0
       aaz0=2.77d0   !2.7d0
c
c       
       qq2=0.d0
c
c     ---------------------------------------------------------------
c     ---------------------------------------------------------------
       integral(1)=0.d0
c
       NDIM2=2
       ACC=1.d-16
       NPRN=1
       IGRAPH=0
       NCALL1=2000
       ITMX1=10
       avgi=0.d0
c       call cuhre(ndim, ncomp, integr, userdata, nvec,
c     &    epsrel, epsabs, verbose + last,
c     &    mineval, maxeval, key,
c     &    statefile, spin,
c     &    nregions, neval, fail, integral, error, prob)
       call suave(ndim, ncomp, integr, userdata, nvec,
     &    epsrel, epsabs, verbose + last, seed,
     &    mineval, maxeval, nnew, nmin, flatness,
     &    statefile, spin,
     &    nregions, neval, fail, integral, error, prob)

       resultado=integral(1)
c       resultado=avgi
       write(*,*) qq2,resultado
       if (fail == 0) then
    	  write(*,*) '  |   Status        CONVERGED        |'
       else
    	  write(*,*) '  |   Status        WARNING: max evals reached    |'
       end if
       
c
      END
c
c
************************************************************************
c
c
	integer function integr(ndim, xx, ncomp, ff)
	implicit none
	integer ndim, ncomp
	double precision xx(*), ff(*)
	double precision A1MAX,A2MAX,JCB,U,V
       double precision mrho2,aa1,aa2,Qt,Qt2,mm,mm0,mm2,integra,Q2
       double precision qq,qq2,Gp_q_k,Gp_q_0,q_inf2,q_sup2
       double precision Umin,Vmin,m_p,m_p2,alfa_sup,alfa_inf
       double precision alfaprop_sup,alfaprop_inf
       double precision lbd,lbd2,g2,be,bep
       double precision Md2,Md2inf,Md2sup
       double precision aa0,aaz0
       double precision pi
	parameter (pi = 3.14159265358979323846D0)
	common/ener/qq2
       common/mass/mm0,aa0,aaz0
       mm=mm0   !FCN=25897.83
       aa1=aa0
       aa2=aaz0
       Qt2=qq2
       Qt=dsqrt(qq2)
       A1MAX=7000.d0
       A2MAX=2.d0*pi
       mm2=mm*mm
       lbd=0.284d0
       lbd2=lbd*lbd
       be=27.d0/(48.d0*pi*pi)   !n_f=3
       bep=27.d0/(12.d0*pi)   !bep=4*pi*be

#define x xx(1)
#define y xx(2)
#define f ff(1)


       U=xx(1)*A1MAX   !U=K
       V=xx(2)*A2MAX   !V=theta
       JCB=A1MAX*A2MAX
c
c
       q_sup2=(Qt2/4.d0)+U*Qt*dcos(V)+U*U
c
       q_inf2=(Qt2/4.d0)-U*Qt*dcos(V)+U*U
c
       Md2=(mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((Qt2+4.d0*mm2)/lbd2))**(1.084d0))    
c
       Md2sup=(mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_sup2+4.d0*mm2)/lbd2))**(1.084d0))
c
       Md2inf=(mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_inf2+4.d0*mm2)/lbd2))**(1.084d0))
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
c
	integr = 0
	end	
c
c
       function integr70(Z)
       implicit none
       double precision pi,mrho2,aa1,aa2,Qt,Qt2,mm,mm2,integr70
       double precision U,V,Q2,Z(1:2)
       double precision ya,yb,yc,yd,ye,yf,yg
       double precision A1MAX,A2MAX,A3MAX,xx
       double precision JCB,avgi,qq,Gp_q_k,Gp_q_0,q_inf2,q_sup2
       double precision Umin,Vmin,m_p,m_p2
       double precision alfaprop_sup,alfaprop_inf
       double precision lbd,lbd2,g2,be,bep
       double precision Md2,Md2inf,Md2sup,aa0,aaz0,mm0,qq2
       integer NDIM2
       common/FER/avgi
       common/dime/NDIM2
       common/ener/qq2
       common/mass/mm0,aa0,aaz0
c
       mm=mm0   !FCN=25897.83
       aa1=aa0
       aa2=aaz0
       pi=4.d0*datan(1.d0)
       Qt2=qq2
       Qt=dsqrt(Qt2)
       A1MAX=7000.d0
       A2MAX=2.d0*pi
       mm2=mm*mm
       lbd=0.284d0
       lbd2=lbd*lbd
       be=27.d0/(48.d0*pi*pi)   !n_f=3
       bep=27.d0/(12.d0*pi)   !bep=4*pi*be
       m_p=0.938272029d0
       m_p2=m_p*m_p
c
       U=Z(1)*A1MAX   !U=K
       V=Z(2)*A2MAX   !V=theta
       JCB=A1MAX*A2MAX
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
       Md2=(mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((Qt2+4.d0*mm2)/lbd2))**(-1.084d0))    
c
       Md2sup=(mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_sup2+4.d0*mm2)/lbd2))**(-1.084d0))
c
       Md2inf=(mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_inf2+4.d0*mm2)/lbd2))**(-1.084d0))
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
       integr70=U*alfaprop_sup*alfaprop_inf*
     *   (Gp_q_0*Gp_q_0-Gp_q_k*(2.d0*Gp_q_0-Gp_q_k))*JCB
c
       else
       integr70=0.d0
       endif
       else
       integr70=0.d0
       endif
c
       return
       end
c  ------------------------------
c       include 'vegasv2.f'
c  ------------------------------       		
