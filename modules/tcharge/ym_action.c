
/*******************************************************************************
*
* File ym_action.c
*
* Copyright (C) 2010-2013, 2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the Yang-Mills action using the symmetric field tensor.
*
*   double ym_action(void)
*     Returns the Yang-Mills action S (w/o prefactor 1/g0^2) of the
*     double-precision gauge field, using a symmetric expression for
*     the gauge-field tensor.
*
*   double ym_action_slices(double *asl)
*     Computes the sum asl[t] of the Yang-Mills action density (w/o
*     prefactor 1/g0^2) of the double-precision gauge field at time
*     t=0,1,...,N0-1 (where N0=NPROC0*L0). The program returns the
*     total action.
*
*   void ym_action_fld(double *f)
*     Assigns the density field s to the observable field f such that
*     f[ix]=s(x), where 0<=ix<VOLUME is the index of the point x.
*
* The Yang-Mills action density s(x) is defined by
*
*  s(x)=(1/4)*sum_{mu,nu} [F_{mu,nu}^a(x)]^2
*
* where
*
*  F_{mu,nu}^a(x)=-2*tr{F_{mu,nu}(x)*T^a}, a=1,..,8,
*
* are the SU(3) components of the symmetric field tensor returned by the
* program ftensor() [ftensor.c]. At the boundaries of the lattice (if any),
* the action density is set to zero. The total action S is the sum of s(x)
* over all points x with time component in the range
*
*  0<x0<NPROC0*L0-1        (open bc),
*
*  0<x0<NPROC0*L0          (SF and open-SF bc),
*
*  0<=x0<NPROC0*L0         (periodic bc).
*
* The programs in this module perform global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define YM_ACTION_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "tcharge.h"
#include "global.h"

#define N0 (NPROC0*L0)

static double *qasl[N0];
static qflt rqasl[N0];
static u3_alg_dble **ft;


static double prodXX(u3_alg_dble *X)
{
   double sm;

   sm=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*X).c1+(*X).c2+(*X).c3)+
      2.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2+(*X).c3*(*X).c3)+
      4.0*((*X).c4*(*X).c4+(*X).c5*(*X).c5+(*X).c6*(*X).c6+
           (*X).c7*(*X).c7+(*X).c8*(*X).c8+(*X).c9*(*X).c9);

   return sm;
}


static double density(int ix)
{
   double sm;

   sm=prodXX(ft[0]+ix)+prodXX(ft[1]+ix)+prodXX(ft[2]+ix)+
      prodXX(ft[3]+ix)+prodXX(ft[4]+ix)+prodXX(ft[5]+ix);

   return sm;
}


double ym_action(void)
{
   int bc,ix,t,tmx;
   double S;

   ft=ftensor();
   bc=bc_type();
   if (bc==0)
      tmx=N0-1;
   else
      tmx=N0;
   qasl[0]=rqasl[0].q;
   qasl[0][0]=0.0;
   qasl[0][1]=0.0;

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (((t>0)&&(t<tmx))||(bc==3))
      {
         S=density(ix);
         acc_qflt(S,qasl[0]);
      }
   }

   if (NPROC>1)
      global_qsum(1,qasl,qasl);

   return 0.5*qasl[0][0];
}


double ym_action_slices(double *asl)
{
   int bc,ix,t,tmx;
   double S;

   ft=ftensor();
   bc=bc_type();
   if (bc==0)
      tmx=N0-1;
   else
      tmx=N0;

   for (t=0;t<N0;t++)
   {
      qasl[t]=rqasl[t].q;
      qasl[t][0]=0.0;
      qasl[t][1]=0.0;
   }

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (((t>0)&&(t<tmx))||(bc==3))
      {
         S=density(ix);
         acc_qflt(S,qasl[t]);
      }
   }

   if (NPROC>1)
      global_qsum(N0,qasl,qasl);

   for (t=0;t<N0;t++)
   {
      asl[t]=0.5*qasl[t][0];

      if (t>0)
         add_qflt(qasl[t],qasl[0],qasl[0]);
   }

   return 0.5*qasl[0][0];
}


void ym_action_fld(double *f)
{
   int bc,ix,t,tmx;

   ft=ftensor();
   bc=bc_type();
   if (bc==0)
      tmx=N0-1;
   else
      tmx=N0;

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (((t>0)&&(t<tmx))||(bc==3))
         f[ix]=0.5*density(ix);
      else
         f[ix]=0.0;
   }
}
