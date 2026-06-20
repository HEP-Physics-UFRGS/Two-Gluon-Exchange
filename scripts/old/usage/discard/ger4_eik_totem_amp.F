      program amplitude_calculator
c --------------------------------------
c      Calculate amplitude for different q values
c      Levels 3,4,5 only - no b, no chi dependencies
c --------------------------------------
      implicit none
      double complex amp_born
      double precision s, sqrt_s, q, tt
      double precision pi, qmin, qmax, qstep
      integer nq, iq
      common/cmener/s,tt
      
      sqrt_s = 7000.d0
      s = sqrt_s*sqrt_s
      tt = 0.05d0          ! Fixed value for tt (if needed)
      pi = 4.d0*datan(1.d0)
      
c     q range
      qmin = 0.01d0
      qmax = 0.2d0
      nq = 20
      qstep = (qmax - qmin)/dble(nq-1)
      
      write(*,*) '======================================='
      write(*,*) 'Amplitude calculation for different q'
      write(*,*) '======================================='
      write(*,*) 'sqrt_s =', sqrt_s, ' GeV'
      write(*,*) 'q range: [', qmin, ',', qmax, ']'
      write(*,*) '======================================='
      write(*,*)
      
c     Loop over q values
      do iq = 1, nq
          q = qmin + dble(iq-1)*qstep
          
          write(*,*) '---------------------------------------'
          write(*,*) 'Computing for q =', q
          
          call compute_amplitude(q, amp_born)
          
          write(*,*) 'amp_born =', amp_born
          write(*,*) '|amp_born| =', cdabs(amp_born)
          write(*,*)
      end do
      
      write(*,*) '======================================='
      write(*,*) 'Calculation completed'
      write(*,*) '======================================='
      
      end program
c
c       
c ====================================================================
c LEVEL 3: Calculate amp_born from diff_t
c ====================================================================
      subroutine compute_amplitude(q, amp_born)
      implicit none
      double precision pi, q
      double precision s, tt
      double precision qq2, be, bep
      double precision mm0, mm2, lbd, lbd2, trajPP, deltaPP
      double precision alfalin
      double precision aa0, aaz0
      double complex imagi, amp_born
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
      double complex diff_t
      integer verbose, nregions, neval, fail
      
      common/cmener/s,tt	
      common/ener/qq2
      common/mass/mm0,aa0,aaz0
      
      pi = 4.d0*datan(1.d0)
      imagi = dcmplx(0.d0,1.d0)
      qq2 = q*q
      be = 27.d0/(48.d0*pi*pi)
      bep = 27.d0/(12.d0*pi)
c
c     Physical parameters
      deltaPP = 0.086557d0
      alfalin = 0.25d0
      mm0 = 0.44137d0
      aa0 = 1.6711d0
      aaz0 = 2.0001d0
c
      integral(1) = 0.d0
c
c ====================================================================
c LEVEL 4: Integration over U,V to get diff_t
c ====================================================================      
      verbose = 0
      call cuhre(ndim, ncomp, integrand, userdata, nvec,
     &    epsrel, epsabs, verbose + last,
     &    mineval, maxeval, key,
     &    statefile, spin,
     &    nregions, neval, fail, integral, error, prob)
      diff_t = integral(1)
      
      write(*,*) '  diff_t =', diff_t
      write(*,*) '  neval =', neval, ' fail =', fail
      
c ====================================================================
c LEVEL 3: Calculate amp_born from diff_t
c ====================================================================
      trajPP = 1.d0 + deltaPP - alfalin*qq2
      amp_born = imagi*(s**trajPP)*8.d0*diff_t
      
      return
      end subroutine
c
c
************************************************************************
c ====================================================================
c LEVEL 5: Integrand - QCD calculations
c ====================================================================      
      integer function integrand(ndim, xx, ncomp, ff)
      implicit none
      integer ndim, ncomp
      double precision xx(*), ff(*)
      double precision A1MAX, A2MAX, JCB, U, V
      double precision aa1, aa2, Qt, Qt2, mm, mm0, mm2
      double precision qq2, Gp_q_k, Gp_q_0, q_inf2, q_sup2
      double precision Umin, Vmin, m_p, m_p2
      double precision alfaprop_sup, alfaprop_inf
      double precision lbd, lbd2, be, bep
      double precision Md2, Md2inf, Md2sup
      double precision aa0, aaz0
      double precision pi, s, tt
      parameter (pi = 3.14159265358979323846D0)
      common/cmener/s,tt
      common/ener/qq2
      common/mass/mm0,aa0,aaz0
      
      mm = mm0
      aa1 = aa0
      aa2 = aaz0
      Qt2 = qq2
      Qt = dsqrt(Qt2)
      mm2 = mm*mm
      lbd = 0.284d0
      lbd2 = lbd*lbd
      be = 27.d0/(48.d0*pi*pi)
      bep = 27.d0/(12.d0*pi)
      m_p = 0.938272029d0
      m_p2 = m_p*m_p
      
#define x xx(1)
#define y xx(2)
#define f ff(1)
      
      A1MAX = s
      A2MAX = 2.d0*pi
      JCB = A1MAX*A2MAX
      U = A1MAX*x
      V = A2MAX*y
      Umin = 0.d0
      Vmin = 0.d0
c	
      Umin = 0.d0
      if (U.ge.Umin) then
c
      Vmin = 0.d0
      if (V.ge.Vmin) then   
c
c     Calculate q_sup and q_inf
      q_sup2 = (Qt2/4.d0) + U*Qt*dcos(V) + U*U
c
      q_inf2 = (Qt2/4.d0) - U*Qt*dcos(V) + U*U
c
c     Calculate dynamic mass scales
      Md2 = (mm2*mm2/(Qt2+mm2))*((dlog(4.d0*mm2/lbd2)/
     *   dlog((Qt2+4.d0*mm2)/lbd2))**(-1.36d0))    
c
      Md2sup = (mm2*mm2/(q_sup2+mm2))*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_sup2+4.d0*mm2)/lbd2))**(-1.36d0))
c
      Md2inf = (mm2*mm2/(q_inf2+mm2))*((dlog(4.d0*mm2/lbd2)/
     *   dlog((q_inf2+4.d0*mm2)/lbd2))**(-1.36d0))
c
c     Calculate running coupling
      alfaprop_sup = 1.d0/(bep*(q_sup2+Md2sup)*
     *  dlog((q_sup2+4.d0*Md2sup)/lbd2))
c
      alfaprop_inf = 1.d0/(bep*(q_inf2+Md2inf)*
     *  dlog((q_inf2+4.d0*Md2inf)/lbd2))
c
c     Calculate proton form factors
      Gp_q_0 = dexp(-aa1*Qt2 - aa2*Qt2*Qt2)
c
      Gp_q_k = dexp(-aa1*(Qt2+9.d0*dabs(U*U-Qt2/4.d0))
     *   - aa2*(Qt2+9.d0*dabs(U*U-Qt2/4.d0))*
     *   (Qt2+9.d0*dabs(U*U-Qt2/4.d0)))
c
c     Final integrand calculation
      f = U*alfaprop_sup*alfaprop_inf*
     *   (Gp_q_0*Gp_q_0 - Gp_q_k*(2.d0*Gp_q_0 - Gp_q_k))*JCB
c
      else
      f = 0.d0
      endif
      else
      f = 0.d0
      endif
c
      integrand = 0
      end
