
/*******************************************************************************
*
* File sw_term.c
*
* Copyright (C) 2011, 2013, 2016, 2018 Martin Luescher, Antonio Rago
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the SW term.
*
*   int sw_order(void)
*     Returns the order N required for the computation of the exponential
*     of the Pauli term [scaled by 1/(4+m0)] to machine precision.
*
*   void pauli_term(double c,u3_alg_dble **ft,pauli_dble *m)
*     Computes the Pauli term using the field tensor ft, multiplies the
*     term by c and assigns the result to m[0] and m[1] (see the notes).
*
*   int sw_term(ptset_t set)
*     Computes the SW term for the current double-precision gauge field
*     and assigns the matrix to the global double-precision SW field. The
*     program inverts the matrices on the specified point set and returns
*     0 if all inversions were safe and 1 if not.
*
* The traditional expression for the SW term is
*
*  c(x0)+csw*(i/4)*sigma_{mu nu}*Fhat_{mu nu}(x),
*
* where
*
*  c(x0) = 4+m0+cF[0]-1     if x0=1 (open, SF or open-SF bc),
*          4+m0+cF[1]-1     if x0=NPROCO*L0-2 (open bc),
*                           or x0=NPROC0*L0-1 (SF or open-SF bc),
*          4+m0             otherwise,
*
*  sigma_{mu nu}=(i/2)*[gamma_mu,gamma_nu],
*
* and Fhat_{mu nu} is the standard (clover) expression for the gauge field
* tensor as computed by the program ftensor() [tcharge/ftensor.c]. The upper
* and lower 6x6 blocks of the matrix are stored in the pauli_dble structures
* swd[2*ix] and swd[2*ix+1], where ix is the label of the point x.
*
* If the alternative "exponential" expression is chosen for the SW term, the
* expression above gets replaced by
*
*  c(x0)*exp{[csw/(4+m0)]*(i/4)*sigma_{mu nu}*Fhat_{mu nu}(x)}.
*
* The quark mass m0, the improvement coefficients csw and cF as well as the
* flag that selects the type of SW term are obtained from the parameter data
* base by calling sw_parms() [flags/lat_parms.c].
*
* Along the boundaries of the lattice at global time
*
*  x0=0                (open, SF and open-SF boundary conditions),
*
*  x0=NPROC0*L0-1      (open boundary conditions),
*
* the SW term is set to unity. Note that the program checks the flags data
* base and computes only those parts of the SW field, which do not already
* have the correct values.
*
* The matrices m[0] and m[1] computed by pauli_term() are the upper and
* lower 6x6 submatrices on the diagonal of the matrix
*
*   -c*(i/2)*sigma_{mu,nu}*Fhat_{mu nu}
*
* at a given lattice point, assuming ft[0],..,ft[5] are the pointers to the
* (0,1),(0,2),(0,3),(2,3),(3,1) components of the field tensor Fhat_{mu nu}
* at this point.
*
* The order N returned by sw_order() is also the one to use when the quark
* forces are calculated using the coefficients returned by sw_dexp().
*
* The programs in this module performs global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define SW_TERM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "tcharge.h"
#include "sw_term.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int N=0;
static double c1,c2,c3[2];
static u3_alg_dble X;
static const pauli_dble sw0={{1.0,1.0,1.0,1.0,1.0,1.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0}};

int sw_order(void)
{
   int n;
   double a,b,c;
   sw_parms_t swp;

   swp=sw_parms();

   if (swp.m0!=DBL_MAX)
   {
      n=0;
      c=3.0*swp.csw/(4.0+swp.m0);
      a=c*exp(c);
      b=DBL_EPSILON;

      for (n=1;n<100;n++)
      {
         a*=c;
         b*=(double)(n+1);

         if (a<b)
            return n;
      }
   }

   error(1,1,"sw_order [swexp.c]","SW parameters are out of range");

   return 0;
}


static void u3_alg2pauli1(pauli_dble *m)
{
   (*m).u[10]=-X.c1;

   (*m).u[12]=-X.c5;
   (*m).u[13]= X.c4;
   (*m).u[14]=-X.c7;
   (*m).u[15]= X.c6;

   (*m).u[18]=-X.c5;
   (*m).u[19]=-X.c4;
   (*m).u[20]=-X.c2;

   (*m).u[22]=-X.c9;
   (*m).u[23]= X.c8;
   (*m).u[24]=-X.c7;
   (*m).u[25]=-X.c6;
   (*m).u[26]=-X.c9;
   (*m).u[27]=-X.c8;
   (*m).u[28]=-X.c3;
}


static void u3_alg2pauli2(pauli_dble *m)
{
   (*m).u[11] =X.c1;
   (*m).u[12]+=X.c4;
   (*m).u[13]+=X.c5;
   (*m).u[14]+=X.c6;
   (*m).u[15]+=X.c7;

   (*m).u[18]-=X.c4;
   (*m).u[19]+=X.c5;

   (*m).u[21] =X.c2;
   (*m).u[22]+=X.c8;
   (*m).u[23]+=X.c9;
   (*m).u[24]-=X.c6;
   (*m).u[25]+=X.c7;
   (*m).u[26]-=X.c8;
   (*m).u[27]+=X.c9;

   (*m).u[29] =X.c3;
}


static void u3_alg2pauli3(pauli_dble *m)
{
   (*m).u[ 0]=-X.c1;
   (*m).u[ 1]=-X.c2;
   (*m).u[ 2]=-X.c3;
   (*m).u[ 3]= X.c1;
   (*m).u[ 4]= X.c2;
   (*m).u[ 5]= X.c3;
   (*m).u[ 6]=-X.c5;
   (*m).u[ 7]= X.c4;
   (*m).u[ 8]=-X.c7;
   (*m).u[ 9]= X.c6;

   (*m).u[16]=-X.c9;
   (*m).u[17]= X.c8;

   (*m).u[30]= X.c5;
   (*m).u[31]=-X.c4;
   (*m).u[32]= X.c7;
   (*m).u[33]=-X.c6;
   (*m).u[34]= X.c9;
   (*m).u[35]=-X.c8;
}


void pauli_term(double c,u3_alg_dble **ft,pauli_dble *m)
{
   _u3_alg_mul_sub(X,c,ft[3][0],ft[0][0]);
   u3_alg2pauli1(m);
   _u3_alg_mul_sub(X,c,ft[4][0],ft[1][0]);
   u3_alg2pauli2(m);
   _u3_alg_mul_sub(X,c,ft[5][0],ft[2][0]);
   u3_alg2pauli3(m);

   m+=1;

   _u3_alg_mul_add(X,c,ft[3][0],ft[0][0]);
   u3_alg2pauli1(m);
   _u3_alg_mul_add(X,c,ft[4][0],ft[1][0]);
   u3_alg2pauli2(m);
   _u3_alg_mul_add(X,c,ft[5][0],ft[2][0]);
   u3_alg2pauli3(m);
}


static int set_swd(int isw,int ofs,int ieo,u3_alg_dble **ft,pauli_dble *sw)
{
   int bc,ix,t,n,ifail;
   double c,*u;
   pauli_dble *sm;

   bc=bc_type();

   if (ofs)
   {
      sw+=2*ofs;
      ft[0]+=ofs;
      ft[1]+=ofs;
      ft[2]+=ofs;
      ft[3]+=ofs;
      ft[4]+=ofs;
      ft[5]+=ofs;
   }

   ifail=0;

   for (ix=0;ix<(VOLUME/2);ix++)
   {
      t=global_time(ix+ofs);

      if (((t==0)&&(bc!=3))||((t==(N0-1))&&(bc==0)))
      {
         sw[0]=sw0;
         sw[1]=sw0;
         sw+=2;
      }
      else
      {
         pauli_term(c2,ft,sw);

         if ((t==1)&&(bc!=3))
            c=c3[0];
         else if (((t==(N0-2))&&(bc==0))||((t==(N0-1))&&((bc==1)||(bc==2))))
            c=c3[1];
         else
            c=c1;

         sm=sw+2;

         for (;sw<sm;sw++)
         {
            if (isw)
            {
               if (ieo)
                  sw_exp(N,ieo,sw,1.0/c,sw);
               else
                  sw_exp(N,ieo,sw,c,sw);
            }
            else
            {
               u=(*sw).u;
               u[0]+=c;
               u[1]+=c;
               u[2]+=c;
               u[3]+=c;
               u[4]+=c;
               u[5]+=c;

               if (ieo)
                  ifail|=inv_pauli_dble(0.0,sw,sw);
            }
         }
      }

      ft[0]+=1;
      ft[1]+=1;
      ft[2]+=1;
      ft[3]+=1;
      ft[4]+=1;
      ft[5]+=1;
   }

   if ((NPROC>1)&&(!isw)&&(ieo))
   {
      MPI_Allreduce(&ifail,&n,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      ifail=n;
   }

   return ifail;
}


static int iswd(int ofs,pauli_dble *sw)
{
   int n,ifail;
   pauli_dble *sm;

   ifail=0;
   sw+=2*ofs;
   sm=sw+VOLUME;

   for (;sw<sm;sw++)
      ifail|=inv_pauli_dble(0.0,sw,sw);

   if (NPROC>1)
   {
      MPI_Allreduce(&ifail,&n,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      ifail=n;
   }

   return ifail;
}


int sw_term(ptset_t set)
{
   int ie,io,isw,ifail,iprms[1];
   pauli_dble *sw;
   u3_alg_dble **ft;
   sw_parms_t swp;
   int my_rank;

   if (NPROC>1)
   {
      iprms[0]=(int)(set);
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=(int)(set),1,"sw_term [sw_term.c]",
            "Parameter is not global");
   }

   swp=sw_parms();

   isw=swp.isw;
   c1=4.0+swp.m0;
   c2=-0.5*swp.csw;
   c3[0]=c1+swp.cF[0]-1.0;
   c3[1]=c1+swp.cF[1]-1.0;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("Operator coefficients:\n");
      printf("m0 = %f\n",swp.m0);
      printf("c1 = 4+m0 = %f\n",c1);
      printf("isw = %d\n", swp.isw);
      printf("csw = %f\n", swp.csw);
      printf("c2 = -0.5*swp.csw = %f\n",c2);
      printf("cF[0] = %f\n",swp.cF[0]);
      printf("cF[1] = %f\n",swp.cF[1]);
      printf("c3[0] = c1+swp.cF[0]-1.0 = 4+m0+swp.cF[0]-1.0 = %f\n", c3[0]);
      printf("c3[1] = c1+swp.cF[0]-1.0 = 4+m0+swp.cF[0]-1.0 = %f\n", c3[1]);
   }

   if (isw)
   {
      N=sw_order();
      c2/=c1;
   }
   else
      N=0;

   sw=swdfld();
   ifail=0;

   if (query_flags(SWD_UP2DATE)!=1)
   {
      ft=ftensor();

      if ((set==NO_PTS)||(set==ODD_PTS))
         (void)(set_swd(isw,0,0,ft,sw));
      else
         ifail|=set_swd(isw,0,1,ft,sw);

      ft=ftensor();

      if ((set==NO_PTS)||(set==EVEN_PTS))
         (void)(set_swd(isw,VOLUME/2,0,ft,sw));
      else
         ifail|=set_swd(isw,VOLUME/2,1,ft,sw);
   }
   else
   {
      ie=query_flags(SWD_E_INVERTED);
      io=query_flags(SWD_O_INVERTED);

      if ((ie==0)&&((set==ALL_PTS)||(set==EVEN_PTS)))
      {
         if (isw)
         {
            ft=ftensor();
            (void)(set_swd(isw,0,1,ft,sw));
         }
         else
            ifail|=iswd(0,sw);
      }

      if ((ie==1)&&((set==NO_PTS)||(set==ODD_PTS)))
      {
         ft=ftensor();
         (void)(set_swd(isw,0,0,ft,sw));
      }

      if ((io==0)&&((set==ALL_PTS)||(set==ODD_PTS)))
      {
         if (isw)
         {
            ft=ftensor();
            (void)(set_swd(isw,VOLUME/2,1,ft,sw));
         }
         else
            ifail|=iswd(VOLUME/2,sw);
      }

      if ((io==1)&&((set==NO_PTS)||(set==EVEN_PTS)))
      {
         ft=ftensor();
         (void)(set_swd(isw,VOLUME/2,0,ft,sw));
      }
   }

   set_flags(COMPUTED_SWD);

   if ((set==ALL_PTS)||(set==EVEN_PTS))
      set_flags(INVERTED_SWD_E);

   if ((set==ALL_PTS)||(set==ODD_PTS))
      set_flags(INVERTED_SWD_O);

   return ifail;
}
