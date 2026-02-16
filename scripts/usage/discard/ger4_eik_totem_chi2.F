	program chi
c --------------------------------------
c      Chi2 calculation program
c --------------------------------------
      implicit none
      double complex chi2,amp_eik,wgauss,resultado
      double precision epsabs,bmin,bmax,s,sqrt_s,b,tt
      double precision pi,diff, diff_t
      external chi2
      common/cmener/s,tt
      
      sqrt_s=7000.d0
      s=sqrt_s*sqrt_s
      epsabs=1d-3
      bmin=0.d0
      bmax=10.d0
      b=0.d0
      tt=0.d0
      pi=4.d0*datan(1.d0)
      
      do
      write(*,*) 'TT:', tt
      
      do
      write(*,*) '  B:', b
      
      resultado=chi2(b)
      write(*,*) '  CHI2:', b, resultado
      
      amp_eik=amp_eik+resultado
      
      b=b+1.d0
      if (b.gt.10.d0) exit
      end do
      
      tt=tt+epsabs
      diff=(cdabs(amp_eik)**2.d0)**0.389379323d0
     *  /(16.d0*pi*s*s)
      
      amp_eik=0.d0
      b=0.d0
      if (tt.gt.0.1d0) exit
      end do
      end program
c
c       
      function chi2(b)
      implicit none
      double precision epsabs,qmin,qmax,s,b,bb,tt
      double complex chi2,chi2_int,wgauss,resultado, chi_val
      COMPLEX(8) :: cd(0:1),j0
      double complex imagi
      external chi2_int
      common/cmener/s,tt
      common/impact/bb
      
      imagi=dcmplx(0.d0,1.d0)
      bb=b
      epsabs=1d-4
      qmin=0.d0
      qmax=0.2d0
      cd(:)=0.d0
      
      if (bb*dsqrt(tt).le.1.d-30) then
      j0=(1.d0,0.d0)
      else
      CALL DBSJA(bb*dsqrt(tt),0.d0,1,10,cd)
      j0=cd(0)
      end if 
      
      chi_val=wgauss(chi2_int,qmin,qmax,epsabs)/s
      chi2=bb*j0*(1.d0-cdexp(imagi*chi_val))
      return
      end 
c
c       
      function chi2_int(q)
      implicit none
      double precision pi,resultado,q
      double precision s,tt,deltass
      double precision deltaq,qq2,alfa0,alfa,g2,be,bep
      double precision mm,mm0,mm2,lbd,lbd2,Md2,trajPP,deltaPP,alfalin
      double precision aa0,aaz0,bb
      double complex imagi,amp_born,chi2_int
      COMPLEX(8) :: cb(0:1),j0
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
      double precision diff_t
      integer verbose, nregions, neval, fail
      
      common/cmener/s,tt	
      common/impact/bb
      common/ener/qq2
      common/mass/mm0,aa0,aaz0
      
      pi=4.d0*datan(1.d0)
      imagi=dcmplx(0.d0,1.d0)
      qq2=q*q
      be=27.d0/(48.d0*pi*pi)
      bep=27.d0/(12.d0*pi)
c
      deltaPP=0.086557d0
      alfalin=0.25d0
      mm0=0.44137d0
      aa0=1.6711d0
      aaz0=2.0001d0
c
      cb(:)=0.d0
      
      if (bb*q.le.1.d-30) then
      j0=(1.d0,0.d0)
      else
      CALL DBSJA(bb*q,0.d0,1,10,cb)
      j0=cb(0)
      end if     
      
      integral(1)=0.d0
c
      
      call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, epsabs, verbose + last,
     &    mineval, maxeval, key,
     &    statefile, spin,
     &    nregions, neval, fail, integral, error, prob)
      diff_t=integral(1)
      
c      write(*,*) '  diff_t = ', diff_t
      trajPP=1.d0+deltaPP-alfalin*qq2
      amp_born=imagi*(s**trajPP)*8.d0*diff_t
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
      double precision aa1,aa2,Qt,Qt2,mm,mm0,mm2,Q2
      double precision qq2,Gp_q_k,Gp_q_0,q_inf2,q_sup2
      double precision Umin,Vmin,m_p,m_p2
      double precision alfaprop_sup,alfaprop_inf
      double precision lbd,lbd2,be,bep
      double precision Md2,Md2inf,Md2sup
      double precision aa0,aaz0
      double precision pi,s,tt
      parameter (pi = 3.14159265358979323846D0)
      common/cmener/s,tt
      common/ener/qq2
      common/mass/mm0,aa0,aaz0
      
      mm=mm0
      aa1=aa0
      aa2=aaz0
      Qt2=qq2
      Qt=dsqrt(Qt2)
      mm2=mm*mm
      lbd=0.284d0
      lbd2=lbd*lbd
      be=27.d0/(48.d0*pi*pi)
      bep=27.d0/(12.d0*pi)
      m_p=0.938272029d0
      m_p2=m_p*m_p
      
#define x xx(1)
#define y xx(2)
#define f ff(1)
      
      A1MAX=s
      A2MAX=2.d0*pi
      JCB=A1MAX*A2MAX
      U=A1MAX*x
      V=A2MAX*y
      Umin=0.d0
      Vmin=0.d0
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
      integrand = 0
      end
