C     integral_T.F
C
C     COMPILACAO:
C       gfortran -O2 -o integral_T integral_T.F -lcuba -lm
C
C     INTEGRAL CALCULADA:
C       I = INT_0^{2pi} dphi INT_0^inf dk
C             k * [T1(k,q2,phi) - T2(k,q2,phi)] * J
C
C     Substituicao de variavel:
C       xx(1)=t -> x=t/(1-t) -> k=sqrt(s)*x   [0,inf)
C       xx(2)=y -> phi=2*pi*y                  [0,2*pi]
C     Jacobiano: J = 2*pi*sqrt(s)/(1-t)^2
C
C     Formulas:
C       m2_log(q2) = mg2*[ln((q2+rho*mg2)/L2)/ln(rho*mg2/L2)]^(-1-g1)
C       Gp(q2)     = exp(-a1*q2 - a2*q2^2)
C       alphaD(q2) = 1/[b0*(q2+m2)*ln((q2+4*m2)/L2)]
C       qkp2 = q2/4 + sqrt(q2)*k*cos(phi) + k^2
C       qkm2 = q2/4 - sqrt(q2)*k*cos(phi) + k^2
C       T1   = alphaD(qkp2)*alphaD(qkm2)*Gp(q2)^2
C       f    = q2 + 9*|k^2 - q2/4|
C       T2   = alphaD(qkp2)*alphaD(qkm2)*Gp(f)*[2*Gp(q2)-Gp(f)]
C
C =================================================================
C     BLOCK DATA: valores dos parametros fisicos
C     Edite os valores DATA abaixo para alterar os parametros
C =================================================================
C
      BLOCK DATA PARAMS
      IMPLICIT NONE
      REAL*8 Q2VAL,MGVAL,LAMVAL,G1VAL,RHOVAL
      REAL*8 A1VAL,A2VAL,B0VAL,SVAL
      COMMON /PHYS/ Q2VAL,MGVAL,LAMVAL,G1VAL,RHOVAL,
     &              A1VAL,A2VAL,B0VAL,SVAL
C
C     -------------------------------------------------------
C     PARAMETROS FISICOS -- edite os valores a seguir
C     -------------------------------------------------------
      DATA Q2VAL  / 0.0D0             /
      DATA MGVAL  / 0.380D0           /
      DATA LAMVAL / 0.284D0           /
      DATA G1VAL  / 0.084D0           /
      DATA RHOVAL / 4.0D0             /
      DATA A1VAL  / 1.491D0           /
      DATA A2VAL  / 2.77D0            /
      DATA B0VAL  / 0.71619724391D0   /
      DATA SVAL   / 13000.0D0           /
C     -------------------------------------------------------
      END
C
C =================================================================
C     PROGRAMA PRINCIPAL
C =================================================================
C
      PROGRAM INTEGT
      IMPLICIT NONE
C
      REAL*8 Q2VAL,MGVAL,LAMVAL,G1VAL,RHOVAL
      REAL*8 A1VAL,A2VAL,B0VAL,SVAL
      COMMON /PHYS/ Q2VAL,MGVAL,LAMVAL,G1VAL,RHOVAL,
     &              A1VAL,A2VAL,B0VAL,SVAL
C
      INTEGER NDIM,NCOMP,NVEC,FLAGS,MINEVAL,MAXEVAL,KEY
      REAL*8  EPSREL,EPSABS
      PARAMETER (NDIM=2, NCOMP=1, NVEC=1)
      PARAMETER (EPSREL=1.0D-8, EPSABS=1.0D-16)
      PARAMETER (FLAGS=0, MINEVAL=100000, MAXEVAL=10000000, KEY=0)
C
      CHARACTER*1 STATEFILE
      INTEGER SPIN
      PARAMETER (STATEFILE=' ')
      DATA SPIN /-1/
C
      INTEGER NREG,NEVAL,FAIL
      REAL*8  INTEG(1),ERR(1),PROB(1)
C
      EXTERNAL INTGDT
C
      CALL CUHRE(NDIM,NCOMP,
     &           INTGDT,
     &           0,
     &           NVEC,
     &           EPSREL,EPSABS,
     &           FLAGS,
     &           MINEVAL,MAXEVAL,
     &           KEY,
     &           STATEFILE,SPIN,
     &           NREG,NEVAL,FAIL,
     &           INTEG,ERR,PROB)
C
      WRITE(*,*)
      WRITE(*,'(A)') '+------------------------------------------+'
      WRITE(*,'(A)') '|   Cuba CUHRE  --  Integral T1 - T2      |'
      WRITE(*,'(A)') '+------------------------------------------+'
      WRITE(*,'(A,F12.6)') '|  q2      = ', Q2VAL
      WRITE(*,'(A,F12.6)') '|  mg      = ', MGVAL
      WRITE(*,'(A,F12.6)') '|  Lambda  = ', LAMVAL
      WRITE(*,'(A,F12.6)') '|  gamma1  = ', G1VAL
      WRITE(*,'(A,F12.6)') '|  rho     = ', RHOVAL
      WRITE(*,'(A,F12.6)') '|  a1      = ', A1VAL
      WRITE(*,'(A,F12.6)') '|  a2      = ', A2VAL
      WRITE(*,'(A,F12.6)') '|  b0      = ', B0VAL
      WRITE(*,'(A,F12.6)') '|  s       = ', SVAL
      WRITE(*,'(A)') '+------------------------------------------+'
      WRITE(*,'(A,G20.15)') '|  Resultado   = ', INTEG(1)
      WRITE(*,'(A,E20.10)') '|  Erro est.   = ', ERR(1)
      WRITE(*,'(A,F12.6)')  '|  Prob. chi2  = ', PROB(1)
      WRITE(*,'(A,I12)')    '|  Avaliacoes  = ', NEVAL
      WRITE(*,'(A,I12)')    '|  Sub-regioes = ', NREG
      IF (FAIL .EQ. 0) THEN
        WRITE(*,'(A)') '|  Status = CONVERGIU                      |'
      ELSE
        WRITE(*,'(A)') '|  Status = AVISO: max. aval. atingido     |'
      END IF
      WRITE(*,'(A)') '+------------------------------------------+'
      WRITE(*,*)
C
      END
C
C =================================================================
C     INTGDT: integrando chamado pelo Cuba
C     Interface exigida pelo Cuba -- nao alterar argumentos
C =================================================================
C
      SUBROUTINE INTGDT(NDIM,XX,NCOMP,FF,USERDT)
      IMPLICIT NONE
C
      INTEGER NDIM,NCOMP,USERDT
      REAL*8  XX(NDIM),FF(NCOMP)
C
      REAL*8 Q2VAL,MGVAL,LAMVAL,G1VAL,RHOVAL
      REAL*8 A1VAL,A2VAL,B0VAL,SVAL
      COMMON /PHYS/ Q2VAL,MGVAL,LAMVAL,G1VAL,RHOVAL,
     &              A1VAL,A2VAL,B0VAL,SVAL
C
      REAL*8 TWOPI
      PARAMETER (TWOPI=6.28318530717958647692D0)
C
      REAL*8 T,X,K,PHI,JAC,SRTS,SRTQ2
      REAL*8 MG2,L2
      REAL*8 QKP2,QKM2
      REAL*8 M2P,M2M
      REAL*8 ALP,ALM
      REAL*8 GPQ2,F,GPF
      REAL*8 T1,T2
C
      T = XX(1)
      IF (T .GE. 1.0D0) THEN
        FF(1) = 0.0D0
        RETURN
      END IF
C
      SRTS  = SQRT(SVAL)
      X     = T / (1.0D0 - T)
      K     = SRTS * X
      PHI   = TWOPI * XX(2)
      JAC   = TWOPI * SRTS / (1.0D0 - T)**2
C
      MG2   = MGVAL  * MGVAL
      L2    = LAMVAL * LAMVAL
      SRTQ2 = SQRT(Q2VAL)
C
      QKP2 = 0.25D0*Q2VAL + SRTQ2*K*COS(PHI) + K*K
      QKM2 = 0.25D0*Q2VAL - SRTQ2*K*COS(PHI) + K*K
      IF (QKP2 .LT. 0.0D0) QKP2 = 0.0D0
      IF (QKM2 .LT. 0.0D0) QKM2 = 0.0D0
C
      CALL EM2LOG(QKP2, MG2, L2, RHOVAL, G1VAL, M2P)
      CALL EM2LOG(QKM2, MG2, L2, RHOVAL, G1VAL, M2M)
C
      CALL ALPHAD(QKP2, M2P, L2, B0VAL, ALP)
      CALL ALPHAD(QKM2, M2M, L2, B0VAL, ALM)
C
      GPQ2 = EXP(-A1VAL*Q2VAL - A2VAL*Q2VAL*Q2VAL)
      F    = Q2VAL + 9.0D0 * ABS(K*K - 0.25D0*Q2VAL)
      GPF  = EXP(-A1VAL*F - A2VAL*F*F)
C
      T1 = ALP * ALM * GPQ2 * GPQ2
      T2 = ALP * ALM * GPF  * (2.0D0*GPQ2 - GPF)
C
      FF(1) = K * (T1 - T2) * JAC
C
      RETURN
      END
C
C =================================================================
C     EM2LOG: m2_log(q2)=mg2*[ln((q2+rho*mg2)/L2)/ln(rho*mg2/L2)]
C                                                      ^(-1-gamma1)
C =================================================================
C
      SUBROUTINE EM2LOG(Q2,MG2,L2,RHO,GAM1,M2OUT)
      IMPLICIT NONE
      REAL*8 Q2,MG2,L2,RHO,GAM1,M2OUT
      REAL*8 ANUM,ADEN,LN1,LN2,RAT
C
      ANUM = Q2 + RHO*MG2
      ADEN = RHO*MG2
C
      IF (ANUM .LE. L2 .OR. ADEN .LE. L2 .OR.
     &    ANUM .LE. 0.0D0 .OR. ADEN .LE. 0.0D0) THEN
        M2OUT = MG2
        RETURN
      END IF
C
      LN1 = LOG(ANUM / L2)
      LN2 = LOG(ADEN / L2)
C
      IF (ABS(LN2) .LT. 1.0D-30) THEN
        M2OUT = MG2
        RETURN
      END IF
C
      RAT = LN1 / LN2
      IF (RAT .LE. 0.0D0) THEN
        M2OUT = MG2
        RETURN
      END IF
C
      M2OUT = MG2 * RAT**(-1.0D0 - GAM1)
C
      RETURN
      END
C
C =================================================================
C     ALPHAD: alpha_D(q2) = 1/[b0*(q2+m2)*ln((q2+4*m2)/L2)]
C =================================================================
C
      SUBROUTINE ALPHAD(Q2,M2,L2,B0,AOUT)
      IMPLICIT NONE
      REAL*8 Q2,M2,L2,B0,AOUT
      REAL*8 ALOG,DEN
C
      ALOG = Q2 + 4.0D0*M2
C
      IF (ALOG .LE. L2 .OR. ALOG .LE. 0.0D0) THEN
        AOUT = 0.0D0
        RETURN
      END IF
C
      DEN = B0 * (Q2 + M2) * LOG(ALOG / L2)
C
      IF (ABS(DEN) .LT. 1.0D-30) THEN
        AOUT = 0.0D0
        RETURN
      END IF
C
      AOUT = 1.0D0 / DEN
C
      RETURN
      END
