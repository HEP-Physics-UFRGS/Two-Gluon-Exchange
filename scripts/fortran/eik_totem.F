c program chi
c --------------------------------------
c      Integral tripla usando VEGAS
c      Variaveis (X,Y,Q2)
c
c   Ver pagina 174, equacao (8.46)
c
c --------------------------------------
c       IMPLICIT DOUBLE PRECISION (A-H,O-Z)
       implicit none
       double complex chi2,amp_eik,wgauss,resultado
       double precision epsabs,bmin,bmax,ss,ssroot,b,tt
       double precision pi,diff
       integer vai
       external chi2
       common/cmener/ss,tt
       vai=101
       ssroot=7000.d0
       ss=ssroot*ssroot
       epsabs=1d-3
       bmin=0.d0
       bmax=10.d0
       b=0.d0
       tt=0.d0
       pi=4.d0*datan(1.d0)
       do
c       amp_eik=wgauss(chi2,bmin,bmax,epsabs)
      	do
      	resultado=chi2(b)
      	amp_eik=amp_eik+resultado
      	write(*,*)"b =",b,"  amp_eik =",amp_eik
      	b=b+1.d0
      	if (b.le.10.d0) then
      	continue
      	else 
      	exit
      	end if
      	end do
      write(*,*)"tt =",tt,"  amp_eik =",amp_eik,
     *  "  ----------------------------------"
      tt=tt+epsabs
      diff=(cdabs(amp_eik)**2.d0)**0.389379323d0
     *  /(16.d0*pi*ss*ss)
      write(vai,*)"tt =",tt,"  diff =",diff
      amp_eik=0.d0
      b=0.d0
      if (tt.le.0.1d0) then
      continue
      else 
      exit
      end if
      end do
c       write(*,*)amp_eik
      end program
c
c       
      function chi2(b)
      implicit none
      double precision epsabs,qmin,qmax,ss,ssroot,b,bb,tt
      double complex amp_born,chi2,chi2_int,wgauss,resultado
      COMPLEX(8) :: cd(0:1),j0
      double complex imagi
      external chi2_int
      common/cmener/ss,tt
      common/impact/bb
      imagi=dcmplx(0.d0,1.d0)
      bb=b
      epsabs=1d-4
      qmin=0.d0
      qmax=0.2d0
      cd(:)=0.d0
!       
      if (bb*dsqrt(tt).le.1.d-30) then
      j0=(1.d0,0.d0)
      else
      CALL DBSJA(bb*dsqrt(tt),0.d0,1,10,cd)
      j0=cd(0)
      continue
      end if 
c       write(*,*)bb
      resultado=wgauss(chi2_int,qmin,qmax,epsabs)/ss
      chi2=bb*j0*(1.d0-cdexp(imagi*resultado))
c       chi2=chi2_int(0.1d0)
c       write(*,*)bb,chi2
      return
      end 
c
c       
      function chi2_int(q)
      double precision pi,resultado,q
      double precision A1MAX,A2MAX,A3MAX,A1MAXt,avgi2,ACC
      double precision sig_tot53,sig_tot70,sig_tot80,sig_tot13
      double precision dif53,dif70,dif80,dif13
      double precision ss,ss53,ss70,ss80,ss13,ssroot0,deltass,tt
      double precision integra,integr53,integr70,integr80,integr13
      double precision deltaq,qq0,qq,qq2,alfa0,alfa,g2,be,bep
      double precision mm,mm0,mm2,lbd,lbd2,Md2,trajPP,deltaPP,alfalin
      double precision aa0,aaz0,sig_tot,bb
      double complex imagi,amplitude,amp_born,chi2_int
      COMPLEX(8) :: cb(0:1),j0
      integer nptq,npts,out1,out2,out3,out4,out5,ieqqe,iesse
      integer ndim2,NCALL1,ITMX1,NPRN,IGRAPH
      integer ndim, ncomp, nvec, last, seed, mineval, maxeval
	double precision epsrel, epsabs, userdata
	parameter (ndim = 2)
	parameter (ncomp = 1)
	parameter (userdata = 0)
	parameter (nvec = 1)
	parameter (epsrel = 1D-3)
	parameter (epsabs = 1D-16)
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
	character*16 env
	integer c
      common/cmener/ss,tt	
      common/impact/bb
      common/ener/qq2
      common/mass/mm0,aa0,aaz0
      pi=4.d0*datan(1.d0)
      imagi=dcmplx(0.d0,1.d0)
      qq2=q*q
      be=27.d0/(48.d0*pi*pi)   !n_f=3
      bep=27.d0/(12.d0*pi)   !bep=4*pi*be
c
c
      deltaPP=0.086557d0   ! Marcela
      alfalin=0.25d0
      mm0=0.44137d0   !0.37965d0
      aa0=1.6711d0   !1.5d0
      aaz0=2.0001d0   !2.7d0
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
c       write(*,*)bb,q,j0
c
c     ---------------------------------------------------------------
c     ---------------------------------------------------------------
      integral(1)=0.d0
c
c
      call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, epsabs, verbose + last,
     &    mineval, maxeval, key,
     &    statefile, spin,
     &    nregions, neval, fail, integral, error, prob)
      resultado=integral(1)
      trajPP=1.d0+deltaPP-alfalin*qq2
      amp_born=imagi*(ss**trajPP)*8.d0*resultado
      chi2_int=q*j0*amp_born
c       write(*,*)q,chi2_int
c
c       npts=80   !25
c       ssroot0=1000.d0
c       deltass=260.d0
c       do iesse=0,npts
c       A1MAXt=ssroot0+iesse*deltass
c       ss=A1MAXt*A1MAXt
c       call cuhre(ndim, ncomp, integr, userdata, nvec,
c     &    epsrel, epsabs, verbose + last,
c     &    mineval, maxeval, key,
c     &    statefile, spin,
c     &    nregions, neval, fail, integral, error, prob)
c       resultado=integral(1)
c       trajPP=1.d0+deltaPP
c       amplitude=imagi*(ss**trajPP)*8.d0*resultado
c       sig_tot=amplitude*0.389379323d0/(imagi*ss)
c       amplitude=0.d0
c       write(out4,*)A1MAXt,sig_tot
ccc       write(*,*)A1MAXt,sig_tot
c
c
c
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
      double precision mrho2,aa1,aa2,Qt,Qt2,mm,mm0,mm2,integra,Q2
      double precision qq,qq2,Gp_q_k,Gp_q_0,q_inf2,q_sup2
      double precision Umin,Vmin,m_p,m_p2,alfa_sup,alfa_inf
      double precision alfaprop_sup,alfaprop_inf
      double precision lbd,lbd2,g2,be,bep
      double precision Md2,Md2inf,Md2sup
      double precision aa0,aaz0
      double precision pi,ss,tt
	parameter (pi = 3.14159265358979323846D0)
	common/cmener/ss,tt
	common/ener/qq2
        common/mass/mm0,aa0,aaz0
        mm=mm0   !FCN=25897.83
      aa1=aa0
      aa2=aaz0
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


	A1MAX=ss
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
