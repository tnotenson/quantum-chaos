! package of routines for fast propagation of density matrices and
! transformations between different representations
!      subroutine init_state(n,nr,tip,tiq,type,phi)
!      subroutine coherent(n,p,q,tiq,tip,sigma,coh)
!      subroutine harstate(n,istate,pr1,pr2,tip,tiq,coh)
!      subroutine state2kirk(n,phi,tip,tiq,rho,ndim,workfft)
!      subroutine prop_init(n,map,pr1,pr2,tip,tiq,rho,nmax,work1,work2)
!      subroutine prop_kirk(n,ktime,a,lda,workfft)
!      subroutine prop_kirk_bak(n,ktime,a,lda,workfft)
!      subroutine diss_init(n,noise,rho,nmax,epsp,epsq)
!      subroutine diss_kirk(n,rho,nmax,workfft)
!      subroutine kirk(idir,n,a,lda,workfft)
!      subroutine kirk2wig(idir,n,a,lda,wig,workfft)
!      subroutine wignerplot(n,wig,ldw,bplot,ndimbplot)
!      subroutine kirk2hus(dist,n,rho,ndim,hus,nr,workfft)
!      subroutine kirk2chord(idir,n,a,lda,nr,chord,ldchord,workfft)
!      complex*16 function psptrace(nn,dist,ndist)
!      real*8 function entropy(nn,dist,ndist)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       subroutine init_state(n,nr,tip,tiq,type,phi)
! initializes various initial states in the coordinate representation
! type can be pos ps2 coh mom , etc
! phi(i) will contain the amplitude <i|phi_0> for i=0,N-1
! states are normalized to 1.
      complex*16 phi(0:n-1)
      complex*16 zero,one, aim,sumc,sigma,zdotc
      character*3 type
      real*8 del,tip,tiq,p,q,anorm,pr1,pr2
      zero=dcmplx(0.d0,0.d0)
      one=dcmplx(1.d0,0.d0)
      aim=dcmplx(0.d0,1.d0)
      del=8.d0*atan2(1.d0,1.d0)/n
      if(type.eq.'pos')then                    !compute <i|iqq>
         write(16,*)' enter q less than 1'
	 read(5,*)q,p
	 iqq=q*n
	 ipp=p*n
	 do i=0,n-1
	    phi(i)=zero
         end do
         phi(iqq)=one
      else if(type.eq.'ps2')then    ! compute 1/sqrt(2){<i|iq1>+ <i|iq2>}
         write(16,*)' enter q1 q2 less than 1'
	 read(5,*)q1,q2
	 iqq=q1*n
	 ipp=q2*n
	 do i=0,n-1
	    phi(i)=zero
         end do
         phi(iqq)=one/sqrt(2.d0)
	 phi(ipp)=one/sqrt(2.d0)

      else if(type.eq.'mom')then              !compute <i|ipp>
         write(16,*)' enter p less than 1'
         read(5,*) q,p
	 iqq=q*n
	 ipp=p*n
	 do i=0,n-1
	    phi(i)= exp(aim*del*(i+tiq)*( ipp +tip))/sqrt(dfloat(n))
	 end do
      else if(type.eq.'coh')then            ! compute <i|ipp,iqq>
         write(16,*)' enter  p,q (less than) 1.'
         read(5,*) q,p
	 sigma=one
	 call coherent(n,p,q,tiq,tip,sigma,phi(0))
!	 anorm=1.d0/dsqrt(dreal(zdotc(n,phi,1,phi,1)))
	 anorm=dsqrt(dble(dot_product(phi(0:n-1),phi(0:n-1))))
!	 call zdscal(n,anorm,phi,1)    ! normalized coherent state
	phi=phi/anorm
      else if (type.eq.'har')then   !diagonalize harper and pick state q
         write(16,*)' enter harper state to be propag.(q less than 1)'
         read(5,*) q, p
	 iqq=n*q
	 pr1=1.d0
	 pr2=1.d0
	 call harstate(n,iqq,pr1,pr2,tip,tiq,phi(0))
      else if (type.eq.'ghz')then
         read(5,*)q,p
         iqq=0
         ipp=n-1
	phi(iqq)=one/sqrt(2.d0)
	phi(ipp)=one/sqrt(2.d0)
!    phi(i) is the #iqq state of the harper hamiltonian
      end if
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine harstate(n,istate,pr1,pr2,tip,tiq,coh)
! computes one eigenvector of the harper hamiltonian for input to init-state
      parameter(ndim=400)
      complex*16 coh(0:n-1),harp(0:ndim-1,0:ndim-1),evec(0:ndim,1)
      real*8 arg,pi,pr1,pr2,tip,tiq,abstol
      real*8 eval(1),rwork(7*ndim)
      complex*16 work(5*ndim)
      dimension iwork(5*ndim),ifail(1)
      if(n.gt.ndim)stop '  dimension N exceeded in harstate  '
      pi=4.d0*atan2(1.d0,1.d0)
      arg=2.d0*pi/n
      do i=0,n-1
         do j=0,n-1
	    harp(i,j)=dcmplx(0.d0,0.d0)
         end do
      end do
      do i=0,n-1
         harp(i,i)=pr1*cos(arg*(i+tiq))
!!Ojo correcciones incluidas el 08/05/2004 porque no compilaba...
!	 harp(i,i+1)=.5d0*pr2*cdexp(0.d0,arg*tip)
      end do
!      harp(0,n-1)=.5d0*pr2*cdexp(0.d0,-arg*tip)
      abstol=2*dlamch('S')
      write(16,*) ' zheevx about to be called'
      call zheevx('V','I','U',n,harp,ndim,1.d0,1.d0,istate,istate   &
      ,abstol,m,eval,evec,ndim,work,2*n,rwork,iwork,ifail,info)
!
	write(16,*) 'harper eigenstate constructed', ifail,info
	write(16,*) istate, '  eigenval.=',eval
	do i=0,n-1
	   coh(i)=evec(i,m)
	end do
!   OBS!!!!!!   avoid the periodicity !!!! (17/12/2003 Ozorio's visit)

!        do i=0,4*m
!	    coh(i)=dcmplx(0.d0,0.d0)
!	    coh(n-1-i)=coh(i)
!	end do
!	do i=0,n-1
!	  write(16,*)i,coh(i)
!	end do
        return
        end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -----------------
      subroutine coherent(n,p,q,tiq,tip,sigma,coh)
! constructs the tiq, tip coherent state in the position representation
! centered at position p, q .le.1
! the state is <i|p,q> a!ording to nonnenmacher thesis (not normalized)
      complex*16 sigma,ex,coh(0:n-1),sumc
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
           coh(i)=sumc
      end do
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine prop_init(n,map,pr1,pr2,tip,tiq,rho,nmax,work1,work2)
! initializes various quantities repeatedly needed in the fast propagation
! utilized by subroutine prop_kirk
! unused rows of rho are used for the purpose
! all subsequent calls should not modify rho(n,*),rho(n+1,*), rho(n+2,*)
      complex*16 rho(0:nmax-1,0:nmax-1),work1(2*n+16),work2(2*n+16)
      complex*16 work3(0:n-1)
      complex*16 twopii
      real*8 pr1,pr2,tip,tiq,twopi,pi,rsum,cc
      character*3 map
      twopii=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))    ! 2*i*pi
      twopi=8.d0*atan2(1.d0,1.d0)                   !2*pi
      pi=twopi/2.d0
      cc=(1.d0+dsqrt(5.d0))/3.d0

!      write(16,*)' from prop_init',n,map,pr1,pr2,tip,tiq
      do i=0,n-1
           rho(n,i)=cdexp(-i*twopii/n)  ! roots of unity for sub. kirk
      end do
      if(map.eq.'har')then       ! this is harper map
      !!sign in exponential changed on 22/12/04
        do j=0,n-1
           rho(n+1,j)=exp(twopii*n*pr1*cos(twopi*(j+tiq)/real(n,8)))
		    end do
		    do j=0,n-1
		        rho(n+2,j)=exp(twopii*n*pr2*cos(twopi*(j+tip)/real(n,8)))
	      end do
      else if(map.eq.'ha2')then       ! this is harper map
         !!sign in exponential changed on 22/12/04
           do j=0,n-1
              rho(n+1,j)=exp(twopii*n*pr1*(-(twopi*(j+tiq)/real(n,8))**2))
            end do
            do j=0,n-1
             rho(n+2,j)=exp(twopii*n*pr2*(-(twopi*(j+tiq)/real(n,8))**2))
            end do
      else if(map.eq.'ha3')then       ! this is harper map
        !!sign in exponential changed on 22/12/04
          do j=0,n-1
             rho(n+1,j)=exp(twopii*cc*n*pr1*cos(twopi*(j+tiq)/(cc*n)))
          end do
          do j=0,n-1
              rho(n+2,j)=exp(twopii*cc*n*pr2*cos(twopi*(j+tip)/(cc*n)))
          end do
      else if(map.eq.'std')then
         ma=(pr1+0.000001)
		      mb=(pr2+0.000001)
      	do j=0,n-1
	   		    rho(n+1,j)=exp(-twopii*n*pr1*cos(twopi*(j+tiq)/real(n,8)))
           !rho(n+1,j)=exp(twopii*n*ma*cos(twopi*(j+tiq)/n))
		    end do
    		do j=0,n-1
    !	   rho(n+2,j)=exp(-twopii*mb*.5d0*(j+tip)**2/n)
    	   		rho(n+2,j)=exp(-twopii*.5d0*(j+tip)**2/n)
        end do
    else if(map.eq.'st3')then
        ma=(pr1+0.000001)
	mb=(pr2+0.000001)
      	do j=0,n-1
	   rho(n+2,j)=exp(twopii*n*pr1*cos(twopi/n*(j+tiq)))
	end do
	do j=0,n-1
	   rho(n+1,j)=exp(-twopii*mb*.5d0*(j+tip)**2/n)
        end do
      else if(map.eq.'st2')then
        ma=(pr1+0.000001)
	mb=(pr2+0.000001)
      	do j=0,n-1
	rho(n+1,j)=exp(twopii*n*pr1*(cos(twopi/n*(j+tiq))+0.7*cos(2.d0*twopi/n*(j+tiq)+twopi/4.d0)))
           !rho(n+1,j)=exp(twopii*n*ma*cos(twopi*(j+tiq)/n))
	end do
	do j=0,n-1
	   rho(n+2,j)=exp(-twopii*mb*.5d0*(j+tip)**2/n)
        end do
  else if(map.eq.'bak')then
      	do j=0,n/2-1
           rho(n+1,j)=exp(twopii/n*j*tip)
	end do
	do j=n/2,n-1
	   rho(n+1,j)=exp(twopii/n*(j-n)*tip)
        end do
      	do j=0,n/2-1
           rho(n+2,j)=exp(twopii/n*j*tiq)
	end do
	do j=n/2,n-1
	   rho(n+2,j)=exp(twopii/n*(j-n)*tiq)
        end do
  else if(map.eq.'ctg')then  ! this is the cat product of two shears
        ma=(2+0.0000001)
	       mb=(1+0.0000001)
      	do j=0,n-1
       !!rho(n+1,j)=exp(-twopii*ma*.5d0*(j+tiq)**2/n)
           rho(n+1,j)=exp( twopii*ma*.5d0*(j+tiq)**2/dfloat(n))        &
                *exp(twopii*n*pr1*cos(twopi/n*(j+tiq)))
	      end do
	       do j=0,n-1
	          !!  rho(n+2,j)=exp(twopii*mb*.5d0*(j+tip)**2/n)
	         rho(n+2,j)=exp(-twopii*mb*.5d0*(j+tip)**2/dfloat(n))  &
                 *exp(twopii*n*pr1*cos(twopi/n*(j+tiq)))
        end do
  else if(map.eq.'int')then  !
	!call init_random_seed()
        ma=(pr2/pr1+0.000001)
        pr2=1.d0
        mb=(1.d0+1.d-12)
        do j=0,n-1
        rho(n+2,j)=exp(-twopii*mb*(j+tip)**2/n)  !
	! call random_number(rnum1)
	! rho(n+2,j)=exp(-twopii*mb*rnum1)
        end do
        do j=0,n-1
!          rho(n+2,j)=exp(-twopii*pr2*(j+tiq)/pr1)
          rho(n+1,j)=exp(twopii*pr2*(j+tiq)/pr1)
         ! write(11,*) rho(n+2,j),mb
        end do

      else if (map.eq.'tra')then ! this is a phase space translation
        ma=(pr1+0.000001)
	       mb=(pr2+0.0000001)
         ! write()
         do j=0,n-1
           rho(n+1,j)=exp(twopii*pr1*(j+tiq))
         ! rho(n+1,j)=exp(twopii*ma*(j+tiq)/n)
	       end do
	       do j=0,n-1
	          ! rho(n+2,j)=exp(-twopii*mb*(j+tip)/n)
     	      rho(n+2,j)=exp(-twopii*pr2*(j+tip))
        end do
      else if (map.eq.'pcg') then ! this is perturbed arnold cat
        ma=1
	       mb=1
      	do j=0,n-1
           rho(n+1,j)=exp(twopii*ma*.5d0*(j+tiq)**2/n)		&
     	          *exp(twopii*n*pr1*cos(twopi/n*(j+tiq)))
	      end do
	      do j=0,n-1
	         rho(n+2,j)=exp(-twopii*mb*.5d0*(j+tip)**2/n)		&
                 *exp(twopii*n*pr2*cos(twopi/n*(j+tip)))
        end do
      else if (map.eq.'pc2') then ! this is perturbed arnold cat
            ma=1
    	       mb=1
          	do j=0,n-1
               rho(n+1,j)=exp(twopii*ma*.5d0*(j+tiq)**2/n)
               ! &
         	     !      *exp(twopii*n*pr1*cos(twopi/n*(j+tiq)))
    	      end do
    	      do j=0,n-1
    	         rho(n+2,j)=exp(-twopii*mb*.5d0*(j+tip)**2/n)		&
                     *exp(twopii*n*pr2*(sin(twopi/n*(j+tip)) - 0.5*sin(2*twopi/n*(j+tip)) ))
            end do
      else if (map.eq.'gro') then  ! this is the grover iteration
        mw=(pr1+0.0001)
	       ms=(pr2+0.0001)
      	do j=0,n-1
           rho(n+1,j)=dcmplx(1.d0,0.d0)
	end do
	rho(n+1,mw)=dcmplx(-1.d0,0.d0)
	do j=0,n-1
	   rho(n+2,j)=dcmplx(-1.d0,0.d0)
        end do
	rho(n+2,ms)=dcmplx(1.d0,0.d0)
  else if(map.eq.'saw')then  ! this is the cat product of two shears
        ma=(pr1+0.000001)
		mb=(pr2+0.000001)
      	do j=0,n-1
	       !!rho(n+1,j)=exp(-twopii*ma*.5d0*(j+tiq)**2/n)
	       rho(n+1,j)=exp(twopii*pr1*.5d0*(j+tiq)**2/n)
		end do
		do j=0,n-1
		 !!  rho(n+2,j)=exp(twopii*mb*.5d0*(j+tip)**2/n)
		  rho(n+2,j)=exp(-twopii*mb*.5d0*(j+tip)**2/n)  !
        end do
   end if
      call zfft2di(n,n,work1)
      call zfft2di(2*n,2*n,work2)

      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
      subroutine prop_kirk(n,ktime,a,lda,workfft)
! propagates the kirkwood matrix  a(k,n)=<k|a|n>/<k|n>   by  kicked
! maps of the type U=exp(i*pr2*T(p) ) exp(-i*pr1*V(q)that quantize
! the classical product of shears
!           p'=p-pr1*V'(q)
!           q'=q - pr2*T'(p')
! upon exit a(k,n) contains  <k|UaU+|n>/<k|n>
! ktime can be positive or negative
! a previous call to prop_init should be made as the two work toghether
      complex*16 a(0:lda-1,0:lda-1),workfft(2*n+16)
!      write(16,*)'from prop_kirk',n,ktime,lda
      if(ktime.eq.0)return
!
      if(ktime.gt.0)then      !propagate the map k times forward
         do kk=1,ktime
           do k=0,n-1
              do i=0,n-1
                a(k,i)=a(k,i)*a(n+1,i)
	             end do
           end do
           call kirk(1,n,a,lda,workfft)
           do i=0,n-1
             do k=0,n-1
               a(i,k)= dconjg(a(n+1,i))*a(i,k)*a(n+2,k)
             end do
           end do
           call kirk(-1,n,a,lda,workfft)
           do k=0,n-1
             do i=0,n-1
               a(k,i)=dconjg(a(n+2,k))*a(k,i)
	     end do
           end do
         end do
      else if (ktime.lt.0) then   !propagate k times backwards
         do kk=1,-ktime
           do k=0,n-1
              do i=0,n-1
                a(k,i)=a(n+2,k)*a(k,i)
	      end do
           end do
           call kirk(1,n,a,lda,workfft)
           do i=0,n-1
             do k=0,n-1
               a(i,k)=a(n+1,i)*a(i,k)*dconjg(a(n+2,k))
             end do
           end do
           call kirk(-1,n,a,lda,workfft)
           do k=0,n-1
             do i=0,n-1
               a(k,i)=a(k,i)*dconjg(a(n+1,i))
	     end do
           end do
         end do
      end if
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
      subroutine prop_kirk_bak(n,ktime,a,lda,workfft)
! propagates the kirkwood representation of a by the baker's map
! on input a=<k|a|n>/<k|n>
! on output a=<k|BaB+|n>/<k|n>
! assumes that the floquet phases are initialized by prop_init as
! a(n+1,j)=exp(twopii/n*j*tip) and a(n+2,j)=exp(twopii/n*j*tiq)
      complex*16 a(0:lda-1,0:lda-1),workfft(2*n+16),twopii,cp,cq
      twopii=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0)/n)
      n2=n/2
      cp=a(n+1,n2)**2   ! exp(2ipi/n*tip)
      cq=a(n+2,n2)**2   ! exp(2ipi/n*tiq)
      do k=1,ktime
        do j=0,n-1
          do i=0,n-1
	     a(i,j)=a(i,j)*exp(-twopii*mod(i*j,n))*a(n+1,j)
	  end do
        end do
        call zfft2di(n,n2,workfft)
        call zfft2d(+1,n,n2,a(0,0),lda,workfft)
        call zfft2d(+1,n,n2,a(0,n2),lda,workfft)
        do j=0,n-1
          do i=0,n-1
	     a(i,j)= dconjg(a(n+1,i))*a(i,j)*a(n+2,j)
	  end do
	end do
	call zfft2di(n2,n,workfft)
        call zfft2d(-1,n2,n,a(0,0),lda,workfft)
        call zfft2d(-1,n2,n,a(n2,0),lda,workfft)
        do i=0,n-1
          do j=0,n-1
	     a(i,j)=dconjg(a(n+2,i))*a(i,j)	&
            *exp(+twopii*mod(i*j,n))*2.d0/(n*n)
	  end do
        end do
! restore workfft
        call zfft2di(n,n,workfft)
      end do
      return
        end

!!***********************************************************
     subroutine diss_init(n,noise,rho,nmax,epsp,epsq)
      complex*16 rho(0:nmax-1,0:nmax-1)
      real*8 epsp,epsq,pi
      character*3 noise
!      write(6,*)' diss_init called, noise=',noise
      pi=4.d0*atan2(1.d0,1.d0)
      if(noise(1:3).eq.'dif')then   ! implement a diffusion noise model
        do i=0,n-1
         rho(n+3,i)=dcmplx(exp(-.5d0*(n*epsq/pi*sin((pi*i)/n))**2),0.d0)   !obs
         rho(n+4,i)=dcmplx(exp(-.5d0*(n*epsp/pi*sin((pi*i)/n))**2),0.d0)
        end do
      else if(noise(1:3).eq.'uni')then   !no noise,equivalent to eps=0., 0.
        do i=0,n-1
          rho(n+3,i)=dcmplx(1.d0,0.d0)
          rho(n+4,i)=dcmplx(1.d0,0.d0)
        end do
      else if(noise(1:3).eq.'dpc')then !  depolarizing channel
        do i=0,n-1
          rho(n+3,i)=dcmplx((1.d0-epsp),0.d0)
	  rho(n+4,i)=dcmplx(1.d0,0.d0)
	end do
      else if (noise(1:3).eq.'pdc')then   ! phase damping channel
        do i=0,n-1
          rho(n+3,i)=dcmplx((1.d0-epsp),0.d0)
	  rho(n+4,i)=dcmplx(1.d0,0.d0)
	end do
	rho(n+3,0)=dcmplx(1.d0,0.d0)
	else if(noise(1:3).eq.'sqr')then
	  nex=Int(1.d0/epsp); ney=Int(1./epsq)
	  If(nex.gt.n/2)nex=n/2
	  If(ney.gt.n/2)ney=n/2
	  rho(n+3,0:n-1)=dcmplx(0.d0,0.d0)
	  rho(n+4,0:n-1)=dcmplx(0.d0,0.d0)
	  Do i=0,nex-1
	    rho(n+3,i)=dcmplx(1.d0,0.d0)
	    rho(n+3,n-1-i)=dcmplx(1.d0,0.d0)
	    rho(n+4,i)=dcmplx(1.d0,0.d0)
	    rho(n+4,n-1-i)=dcmplx(1.d0,0.d0)
	  end do
      end if
      return
      end
!!********************************************
      subroutine diss_init_2(n,noise,rho,nmax,epsp,epsq)
      complex*16 rho(0:nmax-1,0:nmax-1),work3(0:n-1),workfft(0:nmax)
      real*8 epsp,epsq,pi,rsum
      character*3 noise
!      write(16,*)' diss_init called, noise=',noise
      pi=4.d0*datan2(1.d0,1.d0)
      work3=dcmplx(0.d0,0.d0)
      rsum=0.d0
      do i=0,n-1
      	do j=-15,15
      		work3(i)=work3(i)+dexp(-real(i-j*n,8)**2/(2.d0*(n*epsp/(2.d0*pi))**2))
      						!dexp(-real(i-j*n,8)**2/(2.d0*(n*epsp/(pi))**2))
      	end do
     end do
     ! rsum=(dot_product(work3,conjg(work3)))
     ! work3=work3/sqrt(rsum)
     rsum=(dot_product(work3,conjg(work3)))
          write(16,*) '------', rsum
         do  i=0,n-1
                	write(45,*) i,abs(work3(i))
         end do
      call zfft1d(1,n,work3,1.0,workfft)
      rsum=(dot_product(work3,dconjg(work3)))
        !  work3=work3/sqrt(rsum)
      rsum=(dot_product(work3,dconjg(work3)))
             write(16,*) '------', rsum

      if(noise(1:3).eq.'dif')then   ! implement a diffusion noise model
        do i=0,n-1
       !  rho(n+3,i)=dcmplx(exp(-.5d0*(n*epsq/pi*sin((pi*i)/n))**2),0.d0)   !obs
       !  rho(n+4,i)=dcmplx(exp(-.5d0*(n*epsp/pi*sin((pi*i)/n))**2),0.d0)
         rho(n+3,i)=work3(i)/cdabs(work3(0))
         rho(n+4,i)=work3(i)/cdabs(work3(0))
         write(46,*) real(i)/real(n),cdabs(work3(i))/cdabs(work3(0)),exp(-.5d0*(n*epsq/pi*sin((pi*i)/n))**2)
        end do
      else if(noise(1:3).eq.'uni')then   !no noise,equivalent to eps=0., 0.
        do i=0,n-1
          rho(n+3,i)=dcmplx(1.d0,0.d0)
          rho(n+4,i)=dcmplx(1.d0,0.d0)
        end do
      else if(noise(1:3).eq.'dpc')then !  depolarizing channel
        do i=0,n-1
          rho(n+3,i)=dcmplx((1.d0-epsp),0.d0)
	  rho(n+4,i)=dcmplx(1.d0,0.d0)
	end do
      else if (noise(1:3).eq.'pdc')then   ! phase damping channel
        do i=0,n-1
          rho(n+3,i)=dcmplx((1.d0-epsp),0.d0)
	  rho(n+4,i)=dcmplx(1.d0,0.d0)
	end do
	rho(n+3,0)=dcmplx(1.d0,0.d0)
	else if(noise(1:3).eq.'sqr')then
	  nex=Int(1.d0/epsp); ney=Int(1./epsq)
	  If(nex.gt.n/2)nex=n/2
	  If(ney.gt.n/2)ney=n/2
	  rho(n+3,0:n-1)=dcmplx(1.d0,0.d0)
	  rho(n+4,0:n-1)=dcmplx(1.d0,0.d0)
!
!	  Do i=0,nex-1
!	    rho(n+3,i)=dcmplx(1.d0,0.d0)
!	    rho(n+3,n-1-i)=dcmplx(1.d0,0.d0)
!	    rho(n+4,i)=dcmplx(1.d0,0.d0)
!	    rho(n+4,n-1-i)=dcmplx(1.d0,0.d0)
!	  end do
      end if
      return
      end
!!--------------------------------------------------------------------
      subroutine diss_kirk(n,rho,nmax,workfft)
      complex*16 rho(0:nmax-1,0:nmax-1)
      complex*16 workfft(2*n+16),cdum
      real*8 pi
      call zfft2d(1,n,n,rho,nmax,workfft)
      pi=4.d0*datan2(1.d0,1.d0)
      do ip=1,n-1
         do in=0,n-1
        rho(ip,in)=rho(ip,in)*rho(n+3,in)*rho(n+4,ip)/(n*n)
	 end do
      end do
	 do in=1,n-1
	    rho(0,in)=rho(0,in)*rho(n+3,in)*rho(n+4,0)/(n*n)
	 end do
      rho(0,0)=rho(0,0)/(n*n)
! this strange consruction is to force preservation of the norm. It has
! no effect on dif, but allows the depolarizing channel
      call zfft2d(-1,n,n,rho,nmax,workfft)
      !rho(0:n-1,0:n-1)=rho(0:n-1,0:n-1)/n**2
      rho(0:n-1,0:n-1)=rho(0:n-1,0:n-1) !/psptrace(n,rho,nmax)
                        return
      end
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    subroutine channels(n,rho,nmax,cmunu,workfft)
!
    complex*16 cmunu(0:nmax-1,0:nmax-1),workfft(2*n+16)
    complex*16 rho(0:nmax-1,0:nmax-1)
!
      call zfft2d(1,n,n,rho,nmax,workfft)
      do ip=0,n-1
         do in=0,n-1
	    rho(ip,in)=rho(ip,in)*cmunu(in,ip)/(n*n)
	 end do
      end do
      call zfft2d(-1,n,n,rho,nmax,workfft)
      return
      end subroutine channels
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
      subroutine diss_chord(n,rho,nmax)
      complex*16 rho(0:nmax-1,0:nmax-1)
      do ip=0,n-1
         do in=0,n-1
	    rho(ip,in)=rho(ip,in)*rho(n+3,in)*rho(n+4,ip)
	 end do
      end do
      return
      end
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!!
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!
integer function fact(number)
integer number
fact=1
If(number.eq.0)then
	fact=1
else
   	do i=1,number
	fact=fact*i
	end do
end if
return
end function fact
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
real*8 function kaka(l,l0,nu)
integer l,l0,nu,i,j,k,k1,k2
kaka=0.d0
If(l==0)then
kaka=0.d0
else
  do i=1,l0	!l+1,l0
	kaka=kaka+log(real(i,8))
  end do
  do j=1,l	!!l+nu+1,l0+nu
       kaka=kaka-log(real(j,8))	!!+log(real(j,8))
  end do
  do k=1,l0-l
       kaka=kaka-2*log(real(k,8))
  end do
    do k1=1,l0+nu	!!l+nu+1,l0+nu
       kaka=kaka+log(real(k1,8))	!!+log(real(j,8))
  end do
    do k2=1,l+nu	!!l+nu+1,l0+nu
       kaka=kaka-log(real(k2,8))	!!+log(real(j,8))
  end do

  kaka=kaka*0.5d0
end if
return
end function kaka
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
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
       call zfft1d(-1,n,work(0),1.0,workfft)
      do k=0,n-1
         do i=0,n-1
 	    rho(k,i)=work(k)*dconjg(phi(i))*exp(del*i*(k+tip))
	 end do
      end do                !rho(k,i)=<k|phi><phi|i>/<k|i>
      return
      end
!--------------------------------------------------------------------
      subroutine state2pos(n,phi,rho,ndim)
!!
!! build density matrix in the position rep. from state |phi>
!! ouput  is phi_{ij}=<i|phi><phi|j>
      complex(8) phi(0:n-1),rho(0:ndim-1,0:ndim-1)
         do i=0,n-1
       	   do j=0,n-1
	   rho(i,j)=phi(i)*dconjg(phi(j))
	   end do
         end do
	return
      end subroutine state2pos
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!!
subroutine pos2kirk(n,rhopos,ndim,rhokirk,workfft)
!! from the position rep rho(i,j)=<i|phi><phi|j> to the kirkwood rep
!! rho(k,i)=<k|phi><phi|i>/<k|i>=FT1d.rho(i,j)/<k|i>
      complex*16 rhopos(0:ndim-1,0:ndim-1),rhokirk(0:ndim-1,0:ndim-1)
      complex*16 workfft(2*n+16),work(0:n-1)
      complex*16 del
 if(n.gt.15000)stop ' n too large in state2kirk '
 del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
 call zfft1di(n,workfft)
 do i=0,n-1
      work(0:n-1)=rhopos(0:n-1,i)
      call zfft1d(-1,n,work(0),1.0,workfft)
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
 if(n.gt.15000)stop ' n too large in kirk2pos '
 del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
 call zfft1di(n,workfft)
 do i=0,n-1
   do j=0,n-1
    rhokirk(i,j)=rhokirk(i,j)*exp(-del*i*j)
   end do
 end do
 do i=0,n-1
      work(0:n-1)=rhokirk(0:n-1,i)
      call zfft1d(1,n,work(0),1.0,workfft)
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
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``~~~~~~~
!!
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!!
subroutine mom2kirk(n,rhomom,ndim,rhokirk,workfft)
!! from the momentum rep rho(i,j)=<k|phi><phi|n> to the kirkwood rep
!! rho(k,i)=<k|phi><phi|i>/<k|i>=rho(i,j)/<k|i>.FT1d
      complex*16 rhomom(0:ndim-1,0:ndim-1),rhokirk(0:ndim-1,0:ndim-1)
      complex*16 workfft(2*n+16),work(0:n-1)
      complex*16 del
 if(n.gt.15000)stop ' n too large in state2kirk '
 del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
 call zfft1di(n,workfft)
 do i=0,n-1
      work(0:n-1)=rhomom(i,0:n-1)
      call zfft1d(-1,n,work(0),1.0,workfft)
	rhokirk(i,0:n-1)=work(0:n-1)
 end do
!!rhokirk(0:n-1,0:n-1)=rhokirk(0:n-1,0:n-1)/real(n,8)

do ii=0,n-1
  do jj=0,n-1
	rhokirk(ii,jj)=rhokirk(ii,jj)*exp(del*ii*jj)
  end do
end do
 return
end subroutine mom2kirk
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
subroutine kirk2mom(n,rhokirk,ndim,rhomom,workfft)
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!! from the kirkwood rep K(rho)(k,i)=<k|phi><phi|i>/<k|i>
!! to the momentum rep rho(k,n)=<k|phi><phi|n> =K(rho)(k,i).FT1d.<k|i>
      complex*16 rhomom(0:ndim-1,0:ndim-1),rhokirk(0:ndim-1,0:ndim-1)
      complex*16 workfft(2*n+16),work(0:n-1)
      complex*16 del
 if(n.gt.15000)stop ' n too large in kirk2mom '
 del=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))/n     !2*i*pi/N
 call zfft1di(n,workfft)
 do i=0,n-1
   do j=0,n-1
    rhokirk(i,j)=rhokirk(i,j)*exp(-del*i*j)
   end do
 end do
 do i=0,n-1
      work(0:n-1)=rhokirk(i,0:n-1)
      call zfft1d(1,n,work(0),1.0,workfft)
	rhomom(i,0:n-1)=work(0:n-1)
 end do

do ii=0,n-1
  do jj=0,n-1
    rhomom(ii,jj)=rhomom(ii,jj)/real(n,8)  !!division by n normalizes the unnormalized FFT
  end do
end do
do i=0,n-1
   do j=0,n-1
    rhokirk(i,j)=rhokirk(i,j)*exp(del*i*j)
   end do
 end do
! write(16,*) 'kirk2mom done...'
 return
end subroutine kirk2mom
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``~~~~~~~
      subroutine kirk(idir,n,a,lda,workfft)
! goes back and forth between kirkwood and antikirkwood
! idir=1 assumes a(k,n)=<k|a|n>/<k|n> on input and produces a(n,k) on
! output
! idir=-1 assumes a(n,k) and produces a(k,n)
! two su!essive calls should reproduce the original
! notice that the fourier transform should NOT be normalized
! but the output satisfies sum_{n,k}=1
! workfft should be initialized by a call to zfft2di(n,n,workfft)
! it uses as working space the vector a(n,*)
! it is used mainly by prop_kirk to propagate kirkwood matrices by kicked maps
! if a is hermitian then kirk and antikirk are hermitian conjugates.
      complex*16 a(0:lda-1,0:lda-1),workfft(2*n+16),den
      den=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0)/n)   !2*i*pi
      do j=0,n-1
         a(n,j)=cdexp(-j*den)          !n'th roots of unity....
      end do
      if(idir.eq.1)then
        do i=0,n-1
          do k=0,n-1
             a(i,k)=a(i,k)*a(n,mod(i*k,n))
	   end do
        end do
        call zfft2d(1,n,n,a,lda,workfft)
        do i=0,n-1
          do k=0,n-1
             a(i,k)=a(i,k)*a(n,mod(i*k,n))/n
	   end do
        end do
      else if(idir.eq.-1)then
        do i=0,n-1
          do k=0,n-1
             a(i,k)=a(i,k)*dconjg(a(n,mod(i*k,n)))
	   end do
        end do
        call zfft2d(-1,n,n,a,lda,workfft)
        do i=0,n-1
          do k=0,n-1
             a(i,k)=a(i,k)*dconjg(a(n,mod(i*k,n)))/n
	   end do
        end do
      end if
      return
      end
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      subroutine kirk2wig(idir,n,a,lda,nr,wig,ldwig,workfft)
! transforms the kirkwood rep of a to the first irreducible
! quarter of the wigner  function
! on input a is the N*N complex array a(k,n) (unchanged)
! on output wig(ip,iq) is the 2N*2N complex array. When a is hermitian then
! all imaginary parts of wig should be zero. Otherwise, for a
!  unitary map it provides the Weyl representation.
!  nr is set equal to 2 for input to the plotting program
!
! idir=1 implies a is the kirkwood rep
! idir=-1 implies a is the antikirk.
      complex*16 a(0:lda-1,0:lda-1),wig(0:ldwig-1,0:ldwig-1)
      complex*16 ipi,temp,workfft(n+16)
      ipi=dcmplx(0.d0,4.d0*atan2(1.d0,1.d0))     !i*pi
      do ik=0,n-1
         do in=0,n-1
	    wig(ik,in)=a(ik,in)*exp(-4.d0*idir*ipi*mod(ik*in,n)/n)
	 end do
      end do
      call zfft2d(idir,n,n,wig,ldwig,workfft)
      do iq=0,n-1
         do ip=0,n-1
	     temp=wig(iq,ip)*exp(-idir*ipi*mod(ip*iq,2*n)/n)/n
	     wig(iq,ip)=temp
	     wig(iq+n,ip)= temp*(-1)**ip
	     wig(iq,ip+n)= temp*(-1)**iq
	     wig(iq+n,ip+n)= temp*(-1)**(iq+ip)
	 end do
      end do
      nr=2
      return
      end
!!
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      subroutine wig2kirk(idir,n,a,lda,nr,wig,ldwig,workfft)
! take wigner wig  to kirkwood a
! idir=1 implies a is the kirkwood rep
! idir=-1 implies a is the antikirk.
      complex*16 a(0:lda-1,0:lda-1),wig(0:ldwig-1,0:ldwig-1)
      complex*16 ipi,temp,workfft(n+16)
      ipi=dcmplx(0.d0,4.d0*atan2(1.d0,1.d0))     !i*pi
      nr=2
      do iq=0,n-1
         do ip=0,n-1
	     temp=wig(iq,ip)*exp(idir*ipi*mod(ip*iq,2*n)/n)
	     wig(iq,ip)=temp
	 end do
      end do
      call zfft2d(-idir,n,n,wig,ldwig,workfft)
      do ik=0,n-1
         do in=0,n-1
	   a(ik,in)= wig(ik,in)*exp(4.d0*idir*ipi*mod(ik*in,n)/n)
	 end do
      end do


      return
      end
!!!!!!!!!!!!!!!!!!!!!
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
      call zfft2di(n*nr,n*nr,workfft)
      call zfft2d(-1,n*nr,n*nr,hus,nhus,workfft)
      do ip=0,n*nr-1
	 do iq=0,n*nr-1
	     hus(ip,iq)=hus(ip,iq)*(-1)**(ip+iq)/(n*n*nr)    !normalization
         end do
      end do
      call zfft2di(n,n,workfft)
!  hus is the husimi on a grid refined by nr
! notice that first index is momentum and second is coordinate!
      return
      end


!!!!!!!!!!!!!!!!!!!!!!c
      subroutine kirk2chord(idir,n,a,lda,nr,chord,ldchord,workfft)
      complex*16 a(0:lda-1,0:lda-1),workfft(n+15),twopii
      complex*16 chord(0:ldchord-1,0:ldchord-1),temp
      twopii=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))    ! 2*i*pi
      do in=0,n-1
         do ip=0,n-1
	    chord(in,ip)=a(in,ip)
	 end do
      end do
      call zfft2d(1,n,n,chord,ldchord,workfft)
      do ip=0,n-1
         do in=0,n-1
	    temp=chord(ip,in)*exp(twopii*.5*mod(ip*in,2*n))
	    chord(ip,in)=temp
	    chord(ip+n,in)=temp*(-1)**in
	    chord(ip,in+n)=temp*(-1)**ip
	    chord(ip+n,in+n)=temp*(-1)**(ip+in)
	 end do
      end do
      nr=2
!   this definition needs the inversion of either p or q (to be determined)
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
!!!!!!!!!!!!!!!!!!!!!c
      subroutine chord2kirk(idir,n,a,lda,nr,chord,ldchord,workfft)
      complex*16 a(0:lda-1,0:lda-1),workfft(n+15),twopii
      complex*16 chord(0:ldchord-1,0:ldchord-1),temp
      twopii=dcmplx(0.d0,8.d0*atan2(1.d0,1.d0))    ! 2*i*pi

      do ip=0,n-1
         do in=0,n-1
	    temp=chord(ip,in)*exp(-twopii*.5*mod(ip*in,2*n))
	    chord(ip,in)=temp
	    chord(ip+n,in)=temp*(-1)**in
	    chord(ip,in+n)=temp*(-1)**ip
	    chord(ip+n,in+n)=temp*(-1)**(ip+in)
	 end do
      end do
      call zfft2d(-1,n,n,chord,ldchord,workfft)
      nn=n*n
      do in=0,n-1
         do ip=0,n-1
	    a(in,ip)=chord(in,ip)/nn
	 end do
      end do

      nr=2
!   this definition needs the inversion of either p or q (to be determined)
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
      complex*16 function psptrace(nn,dist,ndist)
!! computes the phase space trace of the complex array dist
      complex*16 dist(0:ndist-1,0:ndist-1)
      psptrace=dcmplx(0.d0,0.d0)
      do ip=0,nn-1
          do iq=0,nn-1
	    psptrace=psptrace+dist(ip,iq)
	  end do
      end do
      psptrace=psptrace/nn
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 complex*16 function traza(nn,dist,ndist)
 complex*16 dist(0:ndist-1,0:ndist-1)
 	traza=dcmplx(0.d0,0.d0)
	do ii=0,nn-1
	   traza=traza+dist(ii,ii)
	end do
  return
  end function
!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      real*8 function entropy(nn,dist,ndist)
! computes the linear entropy N*tr(dist**2) associated to the distribution
!  nn is the dimension of the distribution (2n for wig and n*nr for hus)
      complex*16 dist(0:ndist-1,0:ndist-1)
      entropy=0.d0
      do ip=0,nn-1
          do iq=0,nn-1
	    entropy=entropy+dist(ip,iq)*dconjg(dist(ip,iq))
	  end do
      end do
      entropy=entropy/nn
      return
      end function
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!c!
      complex*16 function overlap(n,rho1,ndim1,rho2,ndim2)
      complex*16 rho1(0:ndim1-1,0:ndim1-1),rho2(0:ndim2-1,0:ndim2-1)
      overlap=dcmplx(0.d0,0.d0)
      do ik=0,n-1
          do in=0,n-1
	     overlap=overlap+rho1(ik,in)*dconjg(rho2(ik,in))
	  end do
      end do
      overlap=overlap/n
      return
      end function

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine zfft1di(n,workfft)
!  converts the calls of sgi complib to fftw on MAC OSX
	complex*16 workfft(2*n+16)
      return
      end subroutine zfft1di
!cccccccccccccccccccccccccccccccc
      subroutine zfft1d(dir,n,a,stride,workfft)
! converts the sgi calls to dxml calls' implements non normalized FFT'
       integer dir
       integer*8 plan
       character*1 ty
       complex*16 a(0:n-1),workfft(n),b(1:n)
      INTEGER,PARAMETER:: FFTW_ESTIMATE=64
      INTEGER,PARAMETER:: FFTW_MEASURE=0
	  integer*4 nn(1),HOWMANY,IDIST,ODIST,SIGN
      integer*4 ISTRIDE,OSTRIDE
      integer*4 inembed(1),onembed(1)
	do i=1,n
	   b(i)=a(i-1)
	end do
	nn(1)=n
	inembed(1)=n
    onembed(1)=n
	HOWMANY=1;IDIST=1;ODIST=1
	ISTRIDE=1;OSTRIDE=1
       !call dfftw_plan_dft_1d(plan,n,b,b,        &
       !     	(-1)*dir,FFTW_ESTIMATE)
       CALL DFFTW_PLAN_MANY_DFT(plan,1,nn, HOWMANY,a, 		&
        			inembed, ISTRIDE, IDIST,a,onembed, 		&
        			OSTRIDE, ODIST, dir, FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
       !do i=1,n
       !	  a(i-1)=b(i)
       !end do
      return
      end subroutine zfft1d
!cccccccccccccccccccccccccccccccccccccccccccc
      subroutine zfft2di(n,m,workfft)
      complex*16 workfft(n)
      return
      end subroutine zfft2di
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
      complex*16,intent(inout):: a(0:lda-1,0:lda-1)
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
	stop       "************* BYE *****************"
end if

allocate(work(lda,lda))
work(1:lda,1:lda)=a(0:lda-1,0:lda-1)
  !DFFTW_PLAN_MANY_DFT(PLAN, RANK, N, HOWMANY, IN, INEMBED, ISTRIDE, IDIST,
  !						OUT, ONEMBED, OSTRIDE, ODIST, SIGN, FLAGS)
  !CALL DFFTW_PLAN_MANY_DFT(plan,2,nn, HOWMANY,work,inembed, ISTRIDE, IDIST,		&
  !					work,onembed, OSTRIDE, ODIST, SIGN, FFTW_ESSTIMATE)
  CALL DFFTW_PLAN_MANY_DFT(plan,2,nn, HOWMANY,a, 		&
        			inembed, ISTRIDE, IDIST,a,onembed, &
        			OSTRIDE, ODIST, SIGN, FFTW_ESTIMATE)
 !!! call dfftw_plan_dft_2d(plan,n1,n2,a(0:n1-1,0:n2-1),a(0:n1-1,0:n2-1),(-1)*dir,FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
!a(0:lda-1,0:lda-1)=work(1:lda,1:lda)/sqrt(1.d0*n1*n2)
!!a=a/sqrt(1.d0*n1*n2)
      return
      end subroutine zfft2d
!!cccccccccccccccccccccccccccccccc
     subroutine zfft2d23(dir,n1,n2,a,lda,workfft)
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
      complex*16,intent(inout):: a(0:lda-1,0:lda-1)
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
	stop       "************* BYE *****************"
end if

allocate(work(lda,lda))
work(1:lda,1:lda)=a(0:lda-1,0:lda-1)
  !DFFTW_PLAN_MANY_DFT(PLAN, RANK, N, HOWMANY, IN, INEMBED, ISTRIDE, IDIST,
  !						OUT, ONEMBED, OSTRIDE, ODIST, SIGN, FLAGS)
  !CALL DFFTW_PLAN_MANY_DFT(plan,2,nn, HOWMANY,work,inembed, ISTRIDE, IDIST,		&
  !					work,onembed, OSTRIDE, ODIST, SIGN, FFTW_ESSTIMATE)
  CALL DFFTW_PLAN_MANY_DFT(plan,2,nn, HOWMANY,a, 		&
        			inembed, ISTRIDE, IDIST,a,onembed, &
        			OSTRIDE, ODIST, SIGN, FFTW_ESTIMATE)
 !!! call dfftw_plan_dft_2d(plan,n1,n2,a(0:n1-1,0:n2-1),a(0:n1-1,0:n2-1),(-1)*dir,FFTW_ESTIMATE)
       call dfftw_execute(plan)
       call dfftw_destroy_plan(plan)
!a(0:lda-1,0:lda-1)=work(1:lda,1:lda)/sqrt(1.d0*n1*n2)
a=a/sqrt(1.d0*n1*n2)
      return
      end subroutine zfft2d23
!cccccccccccccccccccccccccccccccccccccccccccccc
      subroutine zprod2d(n,m,a,lda,b,ldb)
      complex*16 a(0:lda-1,0:lda-1) , b(0:ldb-1,0:ldb-1)
      do i=0,n-1
         do j=0,m-1
	    a(i,j)=a(i,j)*b(i,j)
	 end do
      end do
      return
      end subroutine
!cccccccccccccccccccccccccccccccccccccccc
      subroutine zprod1d(n,a,stridea,b,strideb)
      complex*16 a(*),b(*)
      integer stridea,strideb
      do i=0,n-1,stridea
         a(i*stridea)=a(i*stridea)*b(i*strideb)
      end do
      return
      end subroutine
!ccccccccccccccccccccccccccccccccccccccc
subroutine write_dim(dim)
integer dim
open(78,file=".dim")
	write(78,*)dim
close(78)
return
end subroutine
!ccccccccccccccccccccccccccccccccccccccc
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
