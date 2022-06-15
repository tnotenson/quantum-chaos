program otoccat
implicit none
CHARACTER(*), PARAMETER :: archivo="elementos_gato.in"
integer NN,in,i,pe,q,nr,tmax,j,ici,nci,i1,dx,dp,iq,ip,j1
integer nmax,ndist,nx,ny
real T1,T2
real(8) tip,tiq,kk,q0,p0,dkk,pi,twopi,q1,p1,gam,fb1,quasiE
complex(8),allocatable::Ucat(:,:)
complex(8),allocatable::X(:,:),A(:,:),B(:,:),M(:,:),M1(:,:),X0(:,:),M2(:,:),P(:,:)
complex(8),allocatable::eval(:)	  !!eval(0:nm2),tr(0:nm2)
complex(8),allocatable::work1(:),work2(:),workfft(:) !! work1(4*nmax+16),work2(4*nmax+16)  !for zfft
complex(8),allocatable::work3(:)	  !! work3(20*nm2)    ! for zgeev
complex(8),allocatable::vr(:,:)		  !! v(0:nm2-1,0:nm2-1)
complex(8),allocatable::phi(:),rho(:,:),av_abab(:),av_otoc(:),phi2(:),hus(:,:)
real(8)coef,ipr2,iprmom,ipr  !! factor del ancho de la gaussiana
complex(8) norm1,norm2
complex(8) vl,vvr,tr1,tr2,tr3,tr0,tr00,YI,tr11,tr22,tr4,trr1,trr0,trr2,trr3,trr4,tr5
real(8),allocatable::rwork(:) 		  !! rwork(2*nm2)           ! for zgeev
character(3) typo,oto1
character(10)enene,cdx,oto,cdp

YI=dcmplx(0.d0,1.d0)
gam=1.d0/300.d0
pi=4.d0*atan2(1.d0,1.d0)
twopi=2*pi
q0=0.25
p0=0.25
read(5,*)NN,kk  !,coef
nmax=nn+6

ndist=4*nmax
!!aca defino la dimension de maxima de la matriz que va a guardar la husimi husimi 
      if(nmax.lt.86)ndist=4*nmax
      if(nmax.lt.46)ndist=6*nmax
      if(nmax.lt.26)ndist=8*nmax
      if(nmax.lt.16)ndist=10*nmax
      if(nmax.lt.12)ndist=16*nmax
      if(nmax.lt.10)ndist=20*nmax
! read(5,*) oto1,dx,dp
dkk=4.d0/(2*pi*nn)
nr=1
call cene(dx,cdx)
call cene(dp,cdp)
oto=oto1
! kk=0.3*500/dble(NN)
open(77,file="kk.dat")
  write(77,*)kk
close(77)
typo='coh'
tmax=11
allocate(av_abab(tmax),av_otoc(tmax))
allocate(Ucat(1:NN,1:NN),X(1:NN,1:NN),M(1:NN,1:NN),M1(1:NN,1:NN))
allocate(M2(1:NN,1:NN),work1(4*NN+16),work2(2*NN+16),P(NN,NN))
allocate(eval(nn),work3(20*nn),rwork(2*nn))
allocate(phi(NN),phi2(NN),rho(NN,NN),X0(NN,NN),vr(nn,nn))
allocate(hus(ndist,ndist))

allocate(workfft(2*NN+16))
open(88,file='cond_ini.dat')

call init_random_seed()
tip=0.d0; tiq=0.d0 ; nci=15

!!esta es la matriz del mapa
call maUSt(Ucat,kk,NN)

!!aca diaginalizo el mapa
call zgeev('N','v',NN,Ucat,NN,eval,vl,1,vr,NN,work3,20*NN,rwork,in)
  write(*,*)' zgeev exited with info=',in,work3(1)

open(83,file='husimi.dat')

  ipr2=0.d0
ipr=0.d0
iprmom=0.d0
  do i=1,nn
    call cene(i,enene)
    !calculo cuasienergias
    quasiE=datan2(aimag(eval(i)),real(eval(i)))

    phi(1:nn)=vr(:,i)
    !calculo la kirkwood del autoestado i
    call state2kirk(nn,phi,tip,tiq,rho,nmax,work1)
    !calculo la husimi del autoestado 
    call kirk2hus(1,nn,rho,nmax,nr,hus,ndist,work1)
    nx=nn*nr ;ny=nx
    hus=hus/NORMH()
    if(i==1)then
       write(*,*) 'entra al if'
       do pe=1,nx
          write(83, *) (real(hus(pe,j)), j=1,nx)
       end do
    end if
    write(*,*)'done hus',i,NORMH(),1.d0/(1.d0*nx**2),nx
    ipr=ipr+IPRH()
!!!!!!!!!!!!!
    write(*,*) real(eval(i)),aimag(eval(i))
    write(23,*) quasiE 
     
!!!!!!!!!!!

  end do
call flush(23)
call flush(24)
close(83)

ipr=ipr/dfloat(nn)

write(58,*)kk,ipr  

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

close(88)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


contains
function NORMH()
  implicit none
  integer i,j
  real(8) iipp,iprh,normh

  iipp=0.d0
  do i=1,nx
    do j=1,ny
      iipp=iipp+cdabs(hus(i,j))
    end do
  end do
  NORMH=iipp
  return 
end function 
!!!
function IPRH()
  implicit none
  integer i,j
  real(8) iipp,iprh

  iipp=0.d0
  do i=1,nx
    do j=1,ny
      iipp=iipp+cdabs(hus(i,j))**2
    end do
  end do
  iprh=iipp
  return 
end function 
!!!!!!!!!!!!!!!!!!!!!
function FF(i,j,k,n)
  implicit none
  integer i,j,n
  real(8) k,pi,perf,nrr,xi1,xi2
  complex(8) Y,FF

  pi=2.D0*dasin(1.0D0)
  Y=dcmplx(0d0,1d0)
  xi1=0.0d0
  xi2=0.0d0

    FF=cdexp(-Y*2.0d0*pi*                                           &
        (dfloat(i-1)+xi1)*(dfloat(j-1)+xi2)/dfloat(N))              &
        *(dsqrt(1.d0/(dfloat(N))))

  return
  end function
  !!!!!
  subroutine maUSt(U,k,n)
  implicit none
  integer n,np,i,j,l,indice1,indice2
  complex(8) U(n,n),UU(n,n),Y,MM(n,n)
  real(8) k,pi,xi1,xi2

  pi=2.D0*dasin(1.0D0)
  Y=dcmplx(0d0,1d0)
  xi1=0.0d0
  xi2=0.0d0

  do j=1,n
    do i=1,n
    ! UU(i,j)=FF(i,j,k,n)*cdexp(-Y*2*pi*dfloat(n)*k*                  &
    !       dcos(-2d0*pi*(dfloat(i-1)+xi1)/dfloat(n)))
    ! MM(i,j)=dconjg(FF(j,i,k,n))*cdexp(-pi*Y*(i-1+xi1)**2/dfloat(n))
      UU(i,j)=cdexp(-Y*2*pi*dfloat(n)*k*                  &
      dcos(-2d0*pi*(dfloat(i-1)+xi1)/dfloat(n)))*dconjg(FF(j,i,k,n))
      MM(i,j)=cdexp(-pi*Y*(i-1+xi1)**2/dfloat(n))*FF(i,j,k,n)
    end do
  end do

  !!!        MM(1:n,1:n)=matmul(U(1:n,1:n),UU(1:n,1:n))

U=matmul(MM,UU)

  ! do j=1,n
  !   do i=1,n
  !   U(i,j)=(0d0,0d0)
  !     do l=1,n
  !     U(i,j)=U(i,j)+MM(i,l)*UU(l,j)
  !     end do
  !   end do
  ! end do

end subroutine
  subroutine cene(numero,cnum)
    implicit none
    character(*) cnum
    integer numero
    if(numero.lt.10)                                  write(cnum,'(i1)')numero
    if(numero.lt.100.and.numero.ge.10)                write(cnum,'(i2)')numero
    if(numero.lt.1000.and.numero.ge.100)              write(cnum,'(i3)')numero
    if(numero.lt.10000.and.numero.ge.1000)            write(cnum,'(i4)')numero
    if(numero.lt.100000.and.numero.ge.10000)          write(cnum,'(i5)')numero
    if(numero.lt.1000000.and.numero.ge.100000)        write(cnum,'(i6)')numero
    if(numero.lt.10000000.and.numero.ge.1000000)        write(cnum,'(i7)')numero
!
  end subroutine

  subroutine construct_U(uqq,n,kk)
    implicit none
    !!S= (a,b)
    !    (c,d)
    !<q'|U|q>= exp[(i/hbar)(d q'^2-2 q' q +a q^2)/(2 b)]/sqrt(2 pi i hbar b)
    !!
    !! en el toro hbar=1/(2 pi N)
    !! q_i= i/N ; con i=0,..., N-1
    !! Entonces
    !! U_mn=exp[(2 pi i/N)(d n^2-2 n m +a m^2)/(2 b)]*sqrt(N/i b)
    !!
    !!Ozorio
    !!para (2,1)(3,2)
    !! Uqq'=(1/iN)^(1/2) exp[(2 pi i/N)(Q^2-QQ'+Q'2)]
    integer aa,bb,cc,dd,n,ii,jj,i,j,inf,sup
   complex(8) uqq(:,:)
   complex(8) pre,YI,arriba,abajo,tha,twopiI,caca
   real(8) pi,twopi,rnum1,phi,phi2,phi3,gam,rin,pert,perf
   real(8) kk
   call init_random_seed()
   ! open(53,file=archivo)
   ! read(53,*) aa,bb,cc,dd
   ! write(50,*) aa,bb,cc,dd
   ! close(53)
   ! aa=2 ; bb=1; cc=3 ; dd=2
      ! aa=2 ; bb=1; cc=7 ; dd=4
  pi=4.d0*datan2(1.d0,1.d0)
  twopi=2*pi
  YI=dcmplx(0.d0,1.d0)
  inf=int(N*(1-coef)/2.d0)
  sup=int(N*(1+coef)/2.d0)
  uqq=dcmplx(0.d0,0.d0)
  pre=cdsqrt(1.d0/(1.d0*n*YI*bb))
  do i=0,n-1
    ii=i+1
     do j=0,n-1
       jj=j+1
       if(i.ge.inf.and.i.lt.sup)then
         perf= (dd*i**2-2.d0*i*j+aa*j**2)/(2.d0*bb) +             &
              ((kk*N**2)/twopi)*(dsin(twopi*i/N)-dsin(4.d0*pi*i/dble(N))/2.d0)
       else
         perf= (dd*i**2-2.d0*i*j+aa*j**2)/(2.d0*bb)
       end if
       uqq(ii,jj)=pre*cdexp(twopi*YI*perf/dble(n))

       ! write(18,*)ii,jj,real(uqq(ii,jj)),aimag(uqq(ii,jj))
     end do
     ! write(18,*)" "
  end do
  write(*,*) 'uqq building: DONE'

  end subroutine
!!!
subroutine construct_UU(uqq,n)
  implicit none
  integer aa,bb,cc,dd,n,ii,jj,i,j
 complex(8) uqq(:,:)
 complex(8) pre,YI,arriba,abajo,tha,twopiI,caca
 real(8) pi,twopi,rnum1,phi,phi2,phi3,gam,rin,pert,perf
 call init_random_seed()
 aa=0 ; bb=1; cc=3 ; dd=0
    ! aa=2 ; bb=1; cc=7 ; dd=4
pi=4.d0*atan2(1.d0,1.d0)
twopi=2*pi
YI=dcmplx(0.d0,1.d0)


uqq=dcmplx(0.d0,0.d0)
pre=cdsqrt(1.d0/(1.d0*n*YI*bb))
pre=1.d0/dsqrt(dble(n))
do i=0,n-1
  ii=i+1
   do j=0,n-1
     jj=j+1
     perf= (dd*i**2-2.d0*i*j+aa*j**2)/(2.d0*bb) +             &
      ((kk*N**2)/twopi**2)*(dsin(twopi*i/N)-dsin(4.d0*pi*i/dble(N))/2.d0)
     uqq(ii,jj)=pre*cdexp(twopi*YI*perf/dble(n))

     ! write(18,*)ii,jj,real(uqq(ii,jj)),aimag(uqq(ii,jj))
   end do
   ! write(18,*)" "
end do
write(*,*) 'uqq building: DONE'

end subroutine
!!!!!!!!!!
subroutine construct_U2(uqq,n)
  implicit none
  integer aa,bb,cc,dd,n,ii,jj,i,j,n2
 complex(8) uqq(:,:), workfft(2*n+16)
 complex(8),allocatable::upp(:,:)
 complex(8) pre,YI,arriba,abajo,tha,twopiI,caca
 real(8) pi,twopi,rnum1,phi,phi2,phi3,gam,rin,kk,pert
 call init_random_seed()

pi=4.d0*atan2(1.d0,1.d0)
twopi=2*pi
YI=dcmplx(0.d0,1.d0)

allocate(upp(n,n))

uqq=dcmplx(0.d0,0.d0)
upp=dcmplx(0.d0,0.d0)
pre=sqrt(1.d0/(n*YI*bb))
call random_number(rnum1)
do i=0,n-1
  ii=i+1
     uqq(ii,ii)=exp(-2*pi*YI*n*rnum1) !exp(-2*pi*YI*i**2/dble(n))
     upp(ii,ii)=exp(-pi*YI*i**2/dble(n))
end do
! call zfft2d(+1,n,n,upp,n,workfft)
! uqq=upp
! uqq=matmul(uqq,upp)/dble(n)
write(*,*) 'uqq building: DONE'
deallocate(upp)
end subroutine
!!!!!!!!!!!
subroutine construct_W(ww,nn)
  implicit none
  integer nn,i,j
  complex(8) ww(:,:)
  ww=0.d0
  do i=1,nn
     if(i.le.nn/2)ww(i,i)=1
     if(i.gt.nn/2)ww(i,i)=-1
     if(i.le.nn/2)ww(i,i+nn/2)=1
     if(i.ge.nn/2)ww(i,i-nn/2)=1

     ! ww(i,i)=1.d0*(nn/2-i)/abs(nn/2-1)
  end do
  ww=ww/sqrt(2.d0)
end subroutine
subroutine construct_Q(ww,nn)
  implicit none
  integer nn,i,j,xi
  complex(8) ww(:,:)
  ww=0.d0
  xi=nint(nn*q0)
  do i=1,nn
        if(i.ge.(xi-dx).and.i.lt.(xi+dx)) ww(i,i)=1.d0
  end do
  ww=ww/sqrt(2.d0)
end subroutine
!!!!!!!!!!!!!!
subroutine AntiI(ww,nn)
  implicit none
  integer nn,i,j
  complex(8) ww(:,:)
  ww=0.d0
  do i=1,nn
     ww(i,nn-i+1)=1

     ! ww(i,i)=1.d0*(nn/2-i)/abs(nn/2-1)
  end do
  ! ww=ww/sqrt(2.d0)
end subroutine
!!!!!!!!!!
  SUBROUTINE init_random_seed()
    INTEGER :: i, n, clock
    INTEGER, DIMENSION(:), ALLOCATABLE :: seed

    CALL RANDOM_SEED(size = n)
    ALLOCATE(seed(n))

    CALL SYSTEM_CLOCK(COUNT=clock)

    seed = clock + 37 * (/ (i - 1, i = 1, n) /)
    CALL RANDOM_SEED(PUT = seed)

    DEALLOCATE(seed)
  END SUBROUTINE
  function traza(mm,n)
    implicit none
    complex(8) traza,mm(n,n)
    integer n,i

    traza=dcmplx(0.d0,0.d0)
    do i=1,n
      traza=traza+mm(i,i)
    end do
    return
  end function

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine init_state(n,nr,tip,tiq,type,phi)
! initializes various initial states in the coordinate representation
! type can be pos ps2 coh mom , etc
! phi(i) will contain the amplitude <i|phi_0> for i=0,N-1
! states are normalized to 1.
implicit none
integer n,nr,ipp,iqq,i
 complex*16 phi(n)
 complex*16 zero,one,sigma,aim,sumc,zdotc
 character*3 type
 real*8 del,tip,tiq,p,q,anorm,pr1,pr2
 zero=dcmplx(0.d0,0.d0)
 one=dcmplx(1.d0,0.d0)
 aim=dcmplx(0.d0,1.d0)
 del=8.d0*atan2(1.d0,1.d0)/n
 q=q0
 p=p0
 if(type.eq.'pos')then                    !compute <i|iqq>
    write(16,*)' enter q less than 1'
    ! read(5,*)q,p
    iqq=q*n
    ipp=p*n
    phi=zero
    phi(iqq+1)=one
 else if(type.eq.'mom')then              !compute <i|ipp>
    write(16,*)' enter p less than 1'
    ! read(5,*) q,p
    iqq=q*n
    ipp=p*n
    do i=0,n-1
     phi(i+1)= exp(aim*del*(i+tiq)*( ipp+1 +tip))/sqrt(dfloat(n))
    end do
 else if(type.eq.'coh')then            ! compute <i|ipp,iqq>
    write(6,*)' enter  p,q (less than) 1.'
    ! read(5,*) q,p
    sigma=one  !*coef
    call coherent(n,p,q,tiq,tip,sigma,phi)
  anorm=dsqrt(dble(dot_product(phi(1:n),phi(1:n))))
  phi=phi/anorm
  end if
 return
end subroutine
!!!
subroutine coherent(n,p,q,tiq,tip,sigma,coh)
! constructs the tiq, tip coherent state in the position representation
! centered at position p, q .le.1
! the state is <i|p,q> acording to nonnenmacher thesis (not normalized)
integer n,jj,jmax
complex*16 sigma,ex,coh(1:n),sumc
real*8 pi,arg,dx,tip,tiq,p,q
!      data pi /3.1415926535897932d0 /
pi=4.d0*atan2(1.d0,1.d0)
jmax=8
arg=2.d0*pi*n
do i=0,n-1
     sumc=dcmplx(0.d0,0.d0)
     do jj=-jmax,jmax
         dx=jj+q-(i+tiq)/n
         ex=arg*(-.5d0*(dx/sigma)**2 		&
               -dcmplx(0.d0,(p*dx -jj*tip/n)))
         sumc=sumc+cdexp(ex)
     end do
     coh(i+1)=sumc
end do
return
end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine state2pos(n,phi,rho)
!!
!! build density matrix in the position rep. from state |phi>
!! ouput  is phi_{ij}=<i|phi><phi|j>
integer n,i,j
complex(8) phi(n),rho(n,n),tr
   do i=1,n
     do j=1,n
       rho(i,j)=phi(i)*dconjg(phi(j))
     end do
   end do
   ! tr=traza(rho,n)
   ! rho=rho/tr
return
end subroutine state2pos
!!!!!!!!!!!!!!!
subroutine state2pos_inc(n,phi,rho)
!!
!! build density matrix in the position rep. from state |phi>
!! ouput  is phi_{ij}=<i|phi><phi|j>
integer n,i,j
complex(8) phi(n),rho(n,n),tr
rho=dcmplx(0.d0,0.d0)
   do i=1,n
       rho(i,i)=phi(i)*dconjg(phi(i))
   end do
   tr=traza(rho,n)
   rho=rho/tr

return
end subroutine state2pos_inc
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine Uimap(upp,n,nmax,gam)
implicit none
integer nmax,n,i,j
 complex(8) upp(0:nmax-1,0:nmax-1)
 complex(8) pre,imag1,arriba,abajo,tha,twopiI,caca
 real(8) pi,twopi,rsum,rnum1,phi,phi2,phi3,gam,rin
 call init_random_seed()
pi=4.d0*atan2(1.d0,1.d0)
twopi=2*pi
imag1=dcmplx(0.d0,1.d0)
 caca=imag1*twopi

rin=1.d0/real(n,8)
tha=rin*(1.d0 - cmplx(cos(gam*n*twopi),sin(gam*n*twopi)));

rsum=0
upp=dcmplx(0.d0,0.d0)
do i=0,n-1
 call Random_number(rnum1)
!  rnum1=dble(i**2)
	pre=dcmplx(dcos(twopi*rnum1),-dsin(twopi*rnum1))
	arriba=tha*pre
	!!write(35,*)rnum1
	rsum=rsum+twopi*rnum1
   do j=0,n-1
	phi3=(i-j+n*gam)/real(n,8)
	abajo=1.d0 - dcmplx(dcos(twopi*phi3),dsin(twopi*phi3))
	if(abs(abajo)==0) write(16,*)'cero!'
!
	upp(i,j)=arriba/abajo
!
!      if(j.eq.i)upp(i,j)=pre
!	write(36,*)i,j,real(upp(i,j)),aimag(upp(i,j))
   end do
!write(36,*)'   '
end do

write(16,*) 'av ',rsum/real(n)
write(16,*) 'upp building: DONE'
end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end program
!      subroutine zfft1di(n,workfft)
!!  converts the calls of sgi complib to fftw on MAC OSX
!	complex*16 workfft(2*n+16)
!      return
!      end subroutine zfft1di
!cccccccccccccccccccccccccccccccc
subroutine pos2mom(n,phipos,ndim,phimom,workfft)
	implicit none
!! from the position rep rho(i,j)=<i|phi><phi|j> to the kirkwood rep
!! rho(k,i)=<k|phi><phi|i>/<k|i>=FT1d.rho(i,j)/<k|i>
      integer n,ndim,i,ii,jj
      complex(8) phipos(1:ndim),phimom(1:ndim)
      complex(8) workfft(2*n+16),work(0:n-1)
      complex(8) del
      real(8) norma
 if(n.gt.3500)stop ' n too large in state2kirk '
 del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/dble(n)     !2*i*pi/N
 ! call zfft1di(n,workfft)
 do i=0,n-1
      work(0:n-1)=phipos(1:n)
      call zfft1d(-1,n,work(0),1,workfft)
	     phimom(1:n)=work(0:n-1)
 end do
 norma=sum(cdabs(phimom(:))**2)
 phimom=phimom/dsqrt(norma)
 return
end subroutine pos2mom
!!!!!
      subroutine zfft1d(dir,n,a,stride,workfft)
! converts the sgi calls to dxml calls' implements non normalized FFT'
       integer dir,stride
       integer*8 plan
       character*1 ty
       complex(8) workfft(n)
       complex*16 a(1:n) !,b(1:n)
      INTEGER,PARAMETER:: FFTW_ESTIMATE=64
      INTEGER,PARAMETER:: FFTW_MEASURE=0
	  integer*4 nn(1),HOWMANY,IDIST,ODIST,SIGN
      integer*4 ISTRIDE,OSTRIDE
      integer*4 inembed(1),onembed(1)
!	do i=1,n
!	   b(i)=a(i-1)
!	end do
	nn(1)=n
	inembed(1)=n
    onembed(1)=n
	HOWMANY=1;IDIST=1;ODIST=1
	ISTRIDE=1;OSTRIDE=1
       !call dfftw_plan_dft_1d(plan,n,b,b,        &
       !     	(-1)*dir,FFTW_ESTIMATE)
!       write(6,*) "a:",a(0)
       CALL DFFTW_PLAN_MANY_DFT(plan,1,nn, HOWMANY,a, 		&
        			inembed, ISTRIDE, IDIST,a,onembed, 		&
        			OSTRIDE, ODIST, dir, FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
       !do i=1,n
       !	  a(i-1)=b(i)
       !end do
!       write(6,*) "a:",a(0)
      return
      end subroutine zfft1d
!cccccccccccccccccccccccccccccccccccccccccccc
!      subroutine zfft2di(n,m,workfft)
!      complex*16 workfft(n)
!      return
!      end subroutine zfft2di
!ccccccccccccccccccccccccccccccccccccccccccccc
     subroutine zfft2d(dir,n1,n2,a,lda,workfft)
! converts the sgi calls to fftw calls on Mac OSX
!      include'dxmldef.for'
      INTEGER,PARAMETER:: FFTW_ESTIMATE=64
      INTEGER,PARAMETER:: FFTW_MEASURE=0

      integer*4 dir,status,dim,SIGN
      integer*4 nn(2),HOWMANY,IDIST,ODIST
      integer*4 ISTRIDE,OSTRIDE
      integer*4 inembed(2),onembed(2)
       integer*8 plan
      character*1 tyc
      complex*16,intent(inout):: a(lda,lda) !!a(0:lda-1,0:lda-1)
      complex*16 workfft(n1)
      complex(8), allocatable::work(:,:)
nn(1)=n1
nn(2)=n2
SIGN=dir
HOWMANY=1;IDIST=1;ODIST=1
ISTRIDE=1;OSTRIDE=1

inembed(1)=lda;onembed(1)=lda
inembed(2)=lda;onembed(2)=lda
if(n1.gt.lda.or.n2.gt.lda)then
	write(16,*) "dim of work vector too small, sorry"
!	write(6,*) "parametros n1,n2,lda:",n1,n2,lda
	stop       "************* BYE *****************"
end if

allocate(work(0:lda-1,0:lda-1))
work(0:lda-1,0:lda-1)=a(1:lda,1:lda)
  !DFFTW_PLAN_MANY_DFT(PLAN, RANK, N, HOWMANY, IN, INEMBED, ISTRIDE, IDIST,
  !						OUT, ONEMBED, OSTRIDE, ODIST, SIGN, FLAGS)
  ! CALL DFFTW_PLAN_MANY_DFT(plan,2,nn, HOWMANY,work,inembed, ISTRIDE, IDIST,		&
  ! 					work,onembed, OSTRIDE, ODIST, SIGN, FFTW_ESSTIMATE)

  CALL DFFTW_PLAN_MANY_DFT(plan,2,nn, HOWMANY,a, 		&
        			inembed, ISTRIDE, IDIST,a,onembed, &
        			OSTRIDE, ODIST, SIGN, FFTW_ESTIMATE)

 ! call dfftw_plan_dft_2d(plan,n1,n2,a(n1,n2),a(n1,n2),(-1)*dir,FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
! a(1:lda,1:lda)=work(0:lda-1,0:lda-1)

a=a/sqrt(1.d0*n1*n2)
      return
      end subroutine zfft2d
      subroutine pos2kirk(n,rhopos,ndim,rhokirk,workfft)
        !! from the position rep rho(i,j)=<i|phi><phi|j> to the kirkwood rep
        !! rho(k,i)=<k|phi><phi|i>/<k|i>=FT1d.rho(i,j)/<k|i>
              complex*16 rhopos(0:ndim-1,0:ndim-1),rhokirk(0:ndim-1,0:ndim-1)
              complex*16 workfft(2*n+16),work(0:n-1)
              complex*16 del
         if(n.gt.3500)stop ' n too large in state2kirk '
         del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
        !  call zfft1di(n,workfft)
         do i=0,n-1
              work(0:n-1)=rhopos(0:n-1,i)
              call zfft1d(-1,n,work(0),1,workfft)
          rhokirk(0:n-1,i)=work(0:n-1)
         end do
        
        do ii=0,n-1
          do jj=0,n-1
            rhokirk(ii,jj)=rhokirk(ii,jj)*exp(del*ii*jj)
          end do
        end do
         return
        end subroutine pos2kirk
        !!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
        subroutine kirk2pos(n,rhokirk,ndim,rhopos,workfft)
        !!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !! from the position rep rho(i,j)=<i|phi><phi|j> to the kirkwood rep
        !! rho(k,i)=<k|phi><phi|i>/<k|i>=FT1d.rho(i,j)/<k|i>
              complex*16 rhopos(0:ndim-1,0:ndim-1),rhokirk(0:ndim-1,0:ndim-1)
              complex*16 workfft(2*n+16),work(0:n-1)
              complex*16 del
         if(n.gt.3500)stop ' n too large in kirk2pos '
         del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
        !  call zfft1di(n,workfft)
         do i=0,n-1
           do j=0,n-1
            rhokirk(i,j)=rhokirk(i,j)*exp(-del*i*j)
           end do
         end do
         do i=0,n-1
              work(0:n-1)=rhokirk(0:n-1,i)
              call zfft1d(1,n,work(0),1,workfft)
          rhopos(0:n-1,i)=work(0:n-1)
         end do
        
        do ii=0,n-1
          do jj=0,n-1
            rhopos(ii,jj)=rhopos(ii,jj)/real(n,8)  !!division by n normalizes the unnormalized FFT
          end do
        end do
        do i=0,n-1
           do j=0,n-1
            rhokirk(i,j)=rhokirk(i,j)*exp(del*i*j)
           end do
         end do
         return
        end subroutine kirk2pos
        subroutine kirk2hus(idir,n,rho,ndim,nr,hus,nhus,workfft)
          ! assumes workfft has been initialized by prop_init
          ! computes the Husimi distribution from thr Kirkwood array
          ! on input rho is the kirkwood matrix matrix  <k|rho|n>/<k/n> (unchanged)
          ! on output hus is the real array <p,q|rho/p,q> discretized
          !   on a phase space grid (n*nr)times(n*nr)
          ! nr (even) is chosen by the program to provide nice smooth plots
          ! and passed on to the plotting program
          !   for N.gt.50 nr=2 is appropriate
          ! when rho is a pure state this is the Husimi distribution
                complex*16 rho(0:ndim-1,0:ndim-1),aim
                complex*16 workfft(n+16)
                real*8 pi
                complex*16 hus(0:nhus-1,0:nhus-1)
                pi=4.d0*atan2(1.d0,1.d0)     ! /3.1415926535897932d0 /
                aim=dcmplx(0.d0,2*pi/n)    !2*i*pi/n
          !      if(n.gt.ndim+1)stop 'dimension of map too large in kirk2hus'
                nr=2
                if(n.lt.80)nr=4
                if(n.lt.40)nr=6
                if(n.lt.20)nr=8
                if(n.lt.10)nr=10
                if(n.lt.6)nr=16
                if(n.lt.4)nr=20
                if(n*nr.gt.nhus)then
                write(16,*) 'nhus=',nhus
                write(16,*) 'n*nr=',n*nr
                stop ' dimension of hus too large in kirk2hus'
                end if
                nnr2=n*nr/2
                do ik=0,n-1
                   do ip=0,n-1
                hus(ik,ip)=rho(ik,ip)
             end do
                end do
          !      call zfft2di(n,n,workfft)
                call zfft2d(1,n,n,hus,nhus,workfft)
          !  hus now contains the N*N generating function
          
                do iq=0,n*nr-1
             do ip=0,n*nr-1
                 hus(iq,ip)=hus(mod(iq,n),mod(ip,n))
                   end do
                end do
                do iq=0,n*nr-1
             do ip=0,n*nr-1
                 hus(iq,ip)=hus(iq,ip)			&
                    *exp(-.5d0*pi/n*((nnr2-iq)**2+(nnr2-ip)**2))	&
                     *exp(.5d0*aim*mod((ip-nnr2)*(iq-nnr2),2*n))
                   end do
                end do
          !  hus now contains the p,q fourier components of the husimi function
                ! call zfft2di(n*nr,n*nr,workfft)
                call zfft2d(-1,n*nr,n*nr,hus,nhus,workfft)
                do ip=0,n*nr-1
             do iq=0,n*nr-1
                 hus(ip,iq)=hus(ip,iq)*(-1)**(ip+iq)/(n*n*nr)    !normalization
                   end do
                end do
                ! call zfft2di(n,n,workfft)
          !  hus is the husimi on a grid refined by nr
          ! notice that first index is momentum and second is coordinate!
                return
      end subroutine
      subroutine zfft1di(n,workfft)
        !  converts the calls of sgi complib to fftw on MAC OSX
          complex*16 workfft(2*n+16)
              return
              end subroutine zfft1di
    
 !!!!!!!!!!
                subroutine state2kirk(n,phi,tip,tiq,rho,ndim,workfft)
                  !
                  ! constructs the kirkwood representation of the pure state vector phi
                  !
                  ! on entry phi(j)=<j|phi> in the coordinate rep. (unchanged on exit)
                  ! on exit rho(k,n) containd the Kirkwood rep. rho(k,n)=<k|rho|n>/<k|n>
                  ! normalization is such that sum_{k,n}=1
                        complex*16 phi(0:n-1),rho(0:ndim-1,0:ndim-1)
                        complex*16 workfft(2*n+16),del,work(0:n-1)
                        real*8 tip,tiq
                        if(n.gt.15000)stop ' n too large in state2kirk '
                        del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
                        do i=0,n-1
                            work(i)=phi(i)*exp(-del*tip*i)
                        end do
                         call zfft1di(n,workfft)
                         call zfft1d(-1,n,work(0),1,workfft)
                        do k=0,n-1
                           do i=0,n-1
                         rho(k,i)=work(k)*dconjg(phi(i))*exp(del*i*(k+tip))
                     end do
                        end do                !rho(k,i)=<k|phi><phi|i>/<k|i>
                        return
                end subroutine

                subroutine zfft2di(n,m,workfft)
                  complex*16 workfft(n)
                  return
                  end subroutine zfft2di
                !cccccccccccccccc
