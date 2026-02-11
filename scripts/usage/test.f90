program chi
  implicit none

  double precision :: sqrt_s, s, epsabs
  double precision :: bmin, bmax, b, tt, pi
  double complex :: amp_eik, resultado
  double precision :: diff

  sqrt_s = 7000.d0
  s = sqrt_s * sqrt_s
  epsabs = 1.d-3

  bmin = 0.d0
  bmax = 10.d0
  b = 0.d0
  tt = 0.d0

  pi = 4.d0 * datan(1.d0)

  amp_eik = (0.d0, 0.d0)

  do
     do
        resultado = chi2(b)
        amp_eik = amp_eik + resultado

        write(*,*) b, amp_eik

        b = b + 1.d0
        if (b .le. bmax) then
           continue
        else
           exit
        end if
     end do

     write(*,*) tt, amp_eik, '----------------------------------'

     diff = (cdabs(amp_eik)**2.d0)**0.389379323d0
     diff = diff / (16.d0 * pi * s * s)

     tt = tt + epsabs

     amp_eik = (0.d0, 0.d0)
     b = 0.d0

     if (tt .le. 0.1d0) then
        continue
     else
        exit
     end if
  end do

contains

  function chi2(b) result(res)
    double precision, intent(in) :: b
    double complex :: res

    ! Função simples para teste
    res = dcmplx(b, 0.d0)
  end function chi2

end program chi

