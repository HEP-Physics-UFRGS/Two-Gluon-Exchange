       program mindiffeik4
       implicit real*8 (a-h,o-z)
       external FCN,FUTIL
       open(1,FILE='min4_eik_atlas_0_2_log_v4.ini',status='OLD')       
       open(2,FILE='min4_eik_atlas_0_2_log_v4.min')
       call MINTIO(1,2,7)
       call MINUIT(FCN,FUTIL)
       close(1)
       close(2)
       stop
       end
c
       subroutine FCN(NPAR,GRAD,FVAL,XVAL,IFLAG,FUTIL)
       implicit real*8 (a-h,o-z)
       double precision a1,a2,a3,a4,a5,a6,a7,a8
       double precision a9,a10,sig
       double precision x(1500),y(1500),dy(1500)
       double precision FTHEO,CHIAUX,CHISQ,FUTIL
       integer j
       dimension XVAL(NPAR),GRAD(NPAR)
       external FUTIL
       common/inte/j
       if (IFLAG .EQ. 1) then
          k=1
          open(3,FILE='ens_atlas_difc0_2.dat',status='OLD')
10        read(3,*,END=20)x(k),y(k),dy(k)
          k=k+1
          go to 10
20        NDAT=k-1
          close(3)
       endif
       CHISQ=0.d0
       	a1 = 0.09d0
	a2=XVAL(1)   ! mm  → passa a ser o parâmetro 1
	a3=XVAL(2)   ! aa1 → passa a ser o parâmetro 2
	a4 = 0.d0
c       a5=XVAL(5)
c       a6=XVAL(6)
       do j=1,NDAT
        FTHEO=FUTIL(a1,a2,a3,a4,x(j))
        sig=dabs(dy(j))
        CHIAUX=((FTHEO-y(j))/sig)**2.d0
        CHISQ=CHISQ+CHIAUX
        WRITE(*,*)j,a1,a2,a3,a4,CHISQ
       end do
       FVAL=CHISQ
       return
       end
c
c
c
c       
       function FUTIL(a1,a2,a3,a4,x)
       implicit none
       double complex chi2,chi2_int,amp_eik,resultado,fa,fb
       double complex ch12,integral,sum_even,sum_odd
       double complex imagi
       COMPLEX(8) :: cd(0:1),j0
       double precision epsabs,bmin,bmax,ss,ssroot,bb,tt
       double precision qmin,qmax
       double precision h,area,eik,dgauss
       double precision pi,diff
       double precision ya,yb,yc
       double precision a1,a2,a3,a4,x,xx,FUTIL
       integer vai,i,n,j
       external eik
       common/inte/j
       common/cmener/ss,tt
       common/para/ya,yb,yc
c       common/impact/bb
       tt=x
       ya=a1
       yb=a2
       yc=a3
       epsabs=1.0d0
       imagi=dcmplx(0.d0,1.d0)
       bmin=0.d0
       bmax=20.d0
       pi=4.d0*datan(1.d0)
c       write(*,*)j,ya,yb,yc
       
       if (j.le.29) then
       	ssroot=7000.d0
       	ss=ssroot*ssroot
c       
       	amp_eik=imagi*dgauss(eik,bmin,bmax,epsabs) 
       	diff=((cdabs(amp_eik)**2.d0)*pi*0.389379323d0)/(ss*ss)
       	FUTIL=diff
       else if (30.le.j .and. j.le.58) then
       	ssroot=8000.d0
       	ss=ssroot*ssroot
c       
       	amp_eik=imagi*dgauss(eik,bmin,bmax,epsabs) 
       	diff=((cdabs(amp_eik)**2.d0)*pi*0.389379323d0)/(ss*ss)
       	FUTIL=diff      
       else
       	ssroot=13000.d0
       	ss=ssroot*ssroot
c       
       	amp_eik=imagi*dgauss(eik,bmin,bmax,epsabs) 
       	diff=((cdabs(amp_eik)**2.d0)*pi*0.389379323d0)/(ss*ss)
       	FUTIL=diff    
       end if
c       write(*,*)j,tt,diff	     	 
c       write(*,*)tt,amp_eik,diff
       return
       end
c       
c       
       function eik(b)
       double precision pi,resultado,q
       double precision A1MAX,A2MAX,A3MAX,A1MAXt,avgi2,ACC
       double precision sig_tot53,sig_tot70,sig_tot80,sig_tot13
       double precision dif53,dif70,dif80,dif13
       double precision ss,ss53,ss70,ss80,ss13,ssroot0,deltass,tt
       double precision integra,integr53,integr70,integr80,integr13
       double precision deltaq,qq0,qq,qq2,alfa0,alfa,g2,be,bep
       double precision mm,mm0,mm2,lbd,lbd2,Md2,trajPP,deltaPP,alfalin
       double precision aa0,aaz0,sig_tot,bb,b,eik
       double precision ya,yb,yc
       double complex imagi,amplitude,amp_born,chi2
       COMPLEX(8) :: cb(0:1),j0
       integer vai
       integer nptq,npts,out1,out2,out3,out4,out5,ieqqe,iesse
       integer ndim2,NCALL1,ITMX1,NPRN,IGRAPH
       integer ndim, ncomp, nvec, last, seed, mineval, maxeval
	double precision epsrel, epsabs, userdata
	parameter (ndim = 3)
	parameter (ncomp = 1)
	parameter (userdata = 0)
	parameter (nvec = 1)
	parameter (epsrel = 1D-3)
	parameter (epsabs = 1D-16)
	parameter (last = 4)
	parameter (seed = 0)
	parameter (mineval = 0)
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
        integer nnew, nmin
	double precision flatness
	parameter (nnew = 1000)
	parameter (nmin = 2)
	parameter (flatness = 25D0)

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

	external integrand

	double precision integral(ncomp), error(ncomp), prob(ncomp)
	integer verbose, nregions, neval, fail
	character*16 env
	integer c
       common/cmener/ss,tt	
       common/impact/bb
       common/para/ya,yb,yc

       bb=b
       pi=4.d0*datan(1.d0)
       imagi=dcmplx(0.d0,1.d0)
       
       if (bb*dsqrt(tt).le.1.d-30) then
       j0=(1.d0,0.d0)
       else
       CALL DBSJA(bb*dsqrt(tt),0.d0,1,10,cb)
       j0=cb(0)
       continue
       end if
c       write(*,*)bb,j0 
c     ---------------------------------------------------------------
c     ---------------------------------------------------------------
c
c
     
       integral(1)=0.d0
       call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, epsabs, verbose + last,
     &    mineval, maxeval, key,
     &    statefile, spin,
     &    nregions, neval, fail, integral, error, prob)
c        print *, "nregions =", nregions
c	print *, "neval    =", neval
c	print *, "fail     =", fail
c	print '(F20.12," +- ",F20.12,"   p = ",F8.3)',
c     &    (integral(c), error(c), prob(c), c = 1, ncomp)
c       call vegas(ndim, ncomp, integrand, userdata, nvec,
c     &    epsrel, epsabs, verbose, seed,
c     &    mineval, maxeval, nstart, nincrease, nbatch,
c     &    gridno, statefile, spin,
c     &    neval, fail, integral, error, prob)
c        print *, "neval    =", neval
c	print *, "fail     =", fail
c	print '(F20.12," +- ",F20.12,"   p = ",F8.3)',
c     &    (integral(c), error(c), prob(c), c = 1, ncomp)
c       call suave(ndim, ncomp, integrand, userdata, nvec,
c     &    epsrel, epsabs, verbose + last, seed,
c     &    mineval, maxeval, nnew, nmin, flatness,
c     &    statefile, spin,
c     &    nregions, neval, fail, integral, error, prob)
c        print *, "nregions =", nregions
c	print *, "neval    =", neval
c	print *, "fail     =", fail
c	print '(F20.12," +- ",F20.12,"   p = ",F8.3)',
c     &    (integral(c), error(c), prob(c), c = 1, ncomp) 
c       call divonne(ndim, ncomp, integrand, userdata, nvec,
c     &    epsrel, epsabs, verbose, seed,
c     &    mineval, maxeval, key1, key2, key3, maxpass,
c     &    border, maxchisq, mindeviation,
c     &    ngiven, ldxgiven, 0, nextra, 0,
c     &    statefile, spin,
c     &    nregions, neval, fail, integral, error, prob)
c       print *, "nregions =", nregions
c	print *, "neval    =", neval
c	print *, "fail     =", fail
c	print '(F20.12," +- ",F20.12,"   p = ",F8.3)',
c     &    (integral(c), error(c), prob(c), c = 1, ncomp)
       resultado=integral(1)
       eik=ss*bb*j0*
     *   (1.d0-dexp(-1.d0*resultado))
c       write(*,*)bb,resultado
       return
      end 
c
c
************************************************************************

	integer function integrand(ndim, xx, ncomp, ff)
	implicit none
	integer ndim, ncomp
	double precision xx(*), ff(*)
	double precision A1MAX,A2MAX,A3MAX,JCB,U,V,W
       double precision mrho2,aa1,Qt,Qt2,mm,mm0,mm2,integra,Q2
       double precision qq,qq2,Gp_q_k,Gp_q_0,q_inf2,q_sup2
       double precision Umin,Vmin,Wmin,m_p,m_p2,alfa_sup,alfa_inf
       double precision alfaprop_sup,alfaprop_inf
       double precision lbd,lbd2,g2,be,bep,alfalin,amp_born,trajPP
       double precision Md2,Md2inf,Md2sup,deltaPP,qmax
       double precision aa0,aaz0,ya,yb,yc
       double precision pi,ss,tt,bb
        COMPLEX(8) :: cb(0:1),j0
	parameter (pi = 3.14159265358979323846D0)
	common/cmener/ss,tt
	common/impact/bb
	common/para/ya,yb,yc
       deltaPP=0.09d0
       alfalin=0.25d0
       mm=yb          !0.37965d0
       aa1=yc   !1.5d0
       mm2=mm*mm
       lbd=0.284d0
       lbd2=lbd*lbd
       be=27.d0/(48.d0*pi*pi)   !n_f=3
       bep=27.d0/(12.d0*pi)   !bep=4*pi*be
       m_p=0.938272029d0
       m_p2=m_p*m_p
       qmax=1.d0
       

#define x xx(1)
#define y xx(2)
#define z xx(3)
#define f ff(1)


	A1MAX=qmax
	A2MAX=dsqrt(ss)
	A3MAX=2.d0*pi
	JCB=A1MAX*A2MAX*A3MAX
	U=A1MAX*x
	V=A2MAX*y
	W=A3MAX*z
c	
c	
c
       cb(:)=0.d0
!       
       if (bb*U.le.1.d-30) then
       j0=(1.d0,0.d0)
       else
       CALL DBSJA(bb*U,0.d0,1,10,cb)
       j0=cb(0)
       continue
       end if    
c
       q_sup2=(U*U/4.d0)+V*U*dcos(W)+V*V
c
       q_inf2=(U*U/4.d0)-V*U*dcos(W)+V*V
c
       trajPP=1.d0+deltaPP-alfalin*U*U       
c
       Md2=(mm2*mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((U*U+4.d0*mm2)/lbd2))**(1.084d0))    
c
       Md2sup=(mm2*mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_sup2+4.d0*mm2)/lbd2))**(1.084d0))
c
       Md2inf=(mm2*mm2)*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_inf2+4.d0*mm2)/lbd2))**(1.084d0))
c
       alfaprop_sup=1.d0/(bep*(q_sup2+Md2sup)*
     *  dlog((q_sup2+4.d0*Md2sup)/lbd2))
c
       alfaprop_inf=1.d0/(bep*(q_inf2+Md2inf)*
     *  dlog((q_inf2+4.d0*Md2inf)/lbd2))
c
       Gp_q_0=dexp(-aa1*U*U)
c
       Gp_q_k=dexp(-aa1*(U*U+9.d0*dabs(V*V-U*U/4.d0)))
c
       amp_born=((ss**trajPP)/ss)*8.d0*V*alfaprop_sup*alfaprop_inf*
     *   (Gp_q_0*Gp_q_0-Gp_q_k*(2.d0*Gp_q_0-Gp_q_k))
c
       f=U*j0*amp_born*JCB
c       write(*,*)bb,U,V,W,amp_born
c
c
c
	integrand = 0
	end	
