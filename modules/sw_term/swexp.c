
/*******************************************************************************
*
* File swexp.c
*
* Copyright (C) 2018 Antonio Rago, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Exponential of traceless 6x6 Hermitian matrices and related functions.
*
*   void sw_exp(int N,int s,pauli_dble *A,double r,pauli_dble *B)
*     Assigns r*exp(A) (if s=0) or r*exp(-A) (if s!=0) to B. The program
*     assumes A is traceless and calculates the exponential approximately
*     by evaluating the Taylor expansion of the exponential up to (and
*     including) order N. B is set to r times the unit matrix if N<=0. It
*     is permissible to set B=A.
*
*   void sw_dexp(int N,pauli_dble *A,double r,double *q)
*     Computes the coeffient q[6*k+l], k,l=0,..,5, required to calculate
*     the derivatives of r*exp(A) with respect to A (see the notes). The
*     program assumes A is traceless and approximates the exponential by
*     evaluating its Taylor expansion up to (and including) order N. If
*     N<=0, the coefficients are set to 0.
*
* The derivative of exp(A) with respect to a parameter t of A is given by
*
*  sum_{k,l} Q_{kl}*A^k*(dA/dt)*A^l,  Q_{kl}=q[6*k+l],
*
* where q[0],..,q[35] are the coefficients calculated by sw_dexp() (with
* r=1.0) and the indices k,l run from 0 to 5. The matrix Q_{kl} is exactly
* symmetric.
*
* The programs in this module do not perform any communications and can be
* locally called. If SSE (AVX) instructions are used, the Pauli matrices
* must be aligned to a 16 (32) byte boundary.
*
*******************************************************************************/

#define SWEXP_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "sw_term.h"

static int Nmx=0;
static double **ksv,**qsv;
static double **csv,*psv;

#if (defined AVX)
static weyl_dble wsv[6] ALIGNED32;
static pauli_dble Asv[3] ALIGNED32;
#else
static weyl_dble wsv[6] ALIGNED16;
static pauli_dble Asv[3] ALIGNED16;
#endif

static void alloc_arrays(int N)
{
   int k,l;
   double *p;

   if (Nmx>0)
   {
      free(ksv[0]);
      free(ksv);
   }

   ksv=malloc((N+9)*sizeof(*ksv));
   p=malloc(((N*(N+1))/2+9*N-1)*sizeof(*p));
   error_loc((ksv==NULL)||(p==NULL),1,"alloc_arrays [swexp.c]",
             "Unable to allocate auxiliary arrays");

   qsv=ksv+N+1;
   csv=qsv+6;

   for (k=0;k<=N;k++)
   {
      ksv[k]=p;
      p+=(N-k+1);
   }

   for (k=0;k<6;k++)
   {
      qsv[k]=p;
      p+=(N-k+1);
   }

   for (k=0;k<2;k++)
   {
      csv[k]=p;
      p+=(N+1);
   }

   psv=p;

   csv[0][0]=1.0;
   csv[1][0]=1.0;

   for (k=0;k<=N;k++)
   {
      if (k>0)
      {
         csv[0][k]=csv[0][k-1]/(double)(k);
         if (k&0x1)
            csv[1][k]=-csv[0][k];
         else
            csv[1][k]=csv[0][k];
      }

      ksv[k][0]=csv[0][k]/(double)(k+1);

      for (l=1;l<=(N-k);l++)
         ksv[k][l]=ksv[k][l-1]/(double)(k+l+1);
   }

   Nmx=N;
}


static void sw_cpoly(pauli_dble *A)
{
   double tr[5];

   pauli2weyl(A,wsv);
   prod_pauli_mat(A,wsv,wsv);
   weyl2pauli(wsv,Asv);
   prod_pauli_mat(A,wsv,wsv);
   weyl2pauli(wsv,Asv+1);

   tr[0]=tr0_pauli_mat(Asv);
   tr[1]=tr0_pauli_mat(Asv+1);
   tr[2]=tr1_pauli_mat(Asv,Asv);
   tr[3]=tr1_pauli_mat(Asv,Asv+1);
   tr[4]=tr1_pauli_mat(Asv+1,Asv+1);

   psv[0]=(1.0/144.0)*(8.0*tr[1]*tr[1]-24.0*tr[4]
                       +tr[0]*(18.0*tr[2]-3.0*tr[0]*tr[0]));
   psv[1]=(1.0/30.0)*(5.0*tr[0]*tr[1]-6.0*tr[3]);
   psv[2]=(1.0/8.0)*(tr[0]*tr[0]-2.0*tr[2]);
   psv[3]=(-1.0/3.0)*tr[1];
   psv[4]=-0.5*tr[0];
}


static void sw_cayley(int N,double *c,double *q)
{
   int k;
   double q5,*cm;

   if (N>5)
   {
      cm=c;
      c=c+N-6;

      q[0]=c[1];
      q[1]=c[2];
      q[2]=c[3];
      q[3]=c[4];
      q[4]=c[5];
      q[5]=c[6];

      for (;c>=cm;c--)
      {
         q5=q[5];
         q[5]=q[4];
         q[4]=q[3]-q5*psv[4];
         q[3]=q[2]-q5*psv[3];
         q[2]=q[1]-q5*psv[2];
         q[1]=q[0]-q5*psv[1];
         q[0]=c[0]-q5*psv[0];
      }
   }
   else
   {
      for (k=0;k<=5;k++)
      {
         if (k<=N)
            q[k]=c[k];
         else
            q[k]=0.0;
      }
   }
}


static void sw_multi_cayley(int N,double *q)
{
   int k,l,m;

   for (k=0;k<=N;k++)
   {
      m=N-k;
      sw_cayley(m,ksv[k],q);

      if (m>5)
         m=5;

      for (l=0;l<=m;l++)
         qsv[l][k]=q[l];
   }

   for (l=0;l<6;l++)
   {
      sw_cayley(N-l,qsv[l],q);
      q+=6;
   }
}


void sw_exp(int N,int s,pauli_dble *A,double r,pauli_dble *B)
{
   int k;
   double q[6];

   if (N>0)
   {
      if (N>Nmx)
         alloc_arrays(N);

      s=(s!=0);
      sw_cpoly(A);
      sw_cayley(N,csv[s],q);

      for (k=0;k<6;k++)
         q[k]*=r;

      lc3_pauli_mat(q+3,A,Asv,Asv+2);
      prod_pauli_mat(Asv+2,wsv,wsv);
      weyl2pauli(wsv,Asv+1);

      lc3_pauli_mat(q,A,Asv,Asv+2);
      add_pauli_mat(Asv+1,Asv+2,B);
   }
   else
   {
      for (k=0;k<36;k++)
      {
         if (k<6)
            (*B).u[k]=r;
         else
            (*B).u[k]=0.0;
      }
   }
}


void sw_dexp(int N,pauli_dble *A,double r,double *q)
{
   int k,l;

   if (N>0)
   {
      if (N>Nmx)
         alloc_arrays(N);

      sw_cpoly(A);
      sw_multi_cayley(N,q);

      for (k=0;k<6;k++)
      {
         q[6*k+k]*=r;

         for (l=0;l<k;l++)
         {
            q[6*k+l]*=r;
            q[6*l+k]=q[6*k+l];
         }
      }
   }
   else
   {
      for (k=0;k<36;k++)
         q[k]=0.0;
   }
}
