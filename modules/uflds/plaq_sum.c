
/*******************************************************************************
*
* File plaq_sum.c
*
* Copyright (C) 2005-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Calculation of plaquette sums.
*
*   double plaq_sum_dble(int icom)
*     Returns the sum of Re[tr{U(p)}] over all unoriented plaquettes p,
*     where U(p) is the product of the double-precision link variables
*     around p. If icom=1 the global sum of the local sums is returned
*     and otherwise just the local sum. The program assumes that the
*     boundary values of the gauge field are correctly set.
*
*   double plaq_wsum_dble(int icom)
*     Same as plaq_sum_dble(), but giving weight 1/2 to the contribution
*     of the space-like plaquettes at the boundaries of the lattice if
*     boundary conditions of type 0,1 or 2 are chosen.
*
*   double plaq_action_slices(double *asl)
*     Computes the time-slice sums asl[x0] of the tree-level O(a)-improved
*     plaquette action density of the double-precision gauge field. The
*     factor 1/g0^2 is omitted and the time x0 runs from 0 to NPROC0*L0-1.
*     The program returns the total action.
*
* The Wilson plaquette action density is defined so that it converges to the
* Yang-Mills action in the classical continuum limit with a rate proportional
* to a^2. In particular, at the boundaries of the lattice (if there are any),
* the space-like plaquettes are given the weight 1/2 and the contribution of
* a plaquette p in the bulk is 2*Re[tr{1-U(p)}].
*
* The time-slice sum asl[x0] computed by plaq_action_slices() includes the
* full contribution to the action of the space-like plaquettes at time x0 and
* 1/2 of the contribution of the time-like plaquettes at time x0 and x0-1.
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define PLAQ_SUM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "su3fcts.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)

static double *qsm[2*N0];
static qflt rqsmE[N0],rqsmB[N0];
static su3_dble *udb;
static su3_dble wd1 ALIGNED16;
static su3_dble wd2 ALIGNED16;


static double plaq_dble(int n,int ix)
{
   int ip[4];
   double sm;

   plaq_uidx(n,ix,ip);

   su3xsu3(udb+ip[0],udb+ip[1],&wd1);
   su3dagxsu3dag(udb+ip[3],udb+ip[2],&wd2);
   cm3x3_retr(&wd1,&wd2,&sm);

   return sm;
}


static qflt local_plaq_sum_dble(int iw)
{
   int bc,ix,t,n;
   double wp,pa;
   qflt rqsm;

   bc=bc_type();

   if (iw==0)
      wp=1.0;
   else
      wp=0.5;

   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;
   udb=udfld();

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);
      pa=0.0;

      if ((t<(N0-1))||(bc!=0))
      {
         for (n=0;n<3;n++)
            pa+=plaq_dble(n,ix);
      }

      if (((t>0)&&(t<(N0-1)))||(bc==3))
      {
         for (n=3;n<6;n++)
            pa+=plaq_dble(n,ix);
      }
      else if ((t==0)||(bc==0))
      {
         if (bc==1)
            pa+=wp*9.0;
         else
         {
            for (n=3;n<6;n++)
               pa+=wp*plaq_dble(n,ix);
         }
      }
      else
      {
         for (n=3;n<6;n++)
            pa+=plaq_dble(n,ix);

         pa+=wp*9.0;
      }

      acc_qflt(pa,rqsm.q);
   }

   return rqsm;
}


double plaq_sum_dble(int icom)
{
   qflt rqsm;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   rqsm=local_plaq_sum_dble(0);

   if ((icom==1)&&(NPROC>1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}


double plaq_wsum_dble(int icom)
{
   qflt rqsm;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   rqsm=local_plaq_sum_dble(1);

   if ((icom==1)&&(NPROC>1))
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm.q[0];
}


double plaq_action_slices(double *asl)
{
   int bc,ix,t,n;
   double smE,smB;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   bc=bc_type();
   udb=udfld();

   for (t=0;t<N0;t++)
   {
      qsm[t]=rqsmE[t].q;
      rqsmE[t].q[0]=0.0;
      rqsmE[t].q[1]=0.0;

      qsm[t+N0]=rqsmB[t].q;
      rqsmB[t].q[0]=0.0;
      rqsmB[t].q[1]=0.0;
   }

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);
      smE=0.0;
      smB=0.0;

      if ((t<(N0-1))||(bc!=0))
      {
         for (n=0;n<3;n++)
            smE+=(3.0-plaq_dble(n,ix));
      }

      if ((t>0)||(bc!=1))
      {
         for (n=3;n<6;n++)
            smB+=(3.0-plaq_dble(n,ix));
      }

      acc_qflt(smE,rqsmE[t].q);
      acc_qflt(smB,rqsmB[t].q);
   }

   global_qsum(2*N0,qsm,qsm);

   if (bc!=3)
      add_qflt(rqsmE[0].q,rqsmB[0].q,rqsmB[0].q);
   else
   {
      rqsmB[0].q[0]*=2.0;
      rqsmB[0].q[1]*=2.0;
      add_qflt(rqsmE[0].q,rqsmB[0].q,rqsmB[0].q);
      add_qflt(rqsmE[N0-1].q,rqsmB[0].q,rqsmB[0].q);
   }

   if (bc==0)
   {
      for (t=1;t<(N0-1);t++)
      {
         rqsmB[t].q[0]*=2.0;
         rqsmB[t].q[1]*=2.0;
         add_qflt(rqsmE[t].q,rqsmB[t].q,rqsmB[t].q);
         add_qflt(rqsmE[t-1].q,rqsmB[t].q,rqsmB[t].q);
      }

      add_qflt(rqsmE[N0-2].q,rqsmB[N0-1].q,rqsmB[N0-1].q);
   }
   else
   {
      for (t=1;t<N0;t++)
      {
         rqsmB[t].q[0]*=2.0;
         rqsmB[t].q[1]*=2.0;
         add_qflt(rqsmE[t].q,rqsmB[t].q,rqsmB[t].q);
         add_qflt(rqsmE[t-1].q,rqsmB[t].q,rqsmB[t].q);
      }
   }

   for (t=0;t<N0;t++)
   {
      asl[t]=rqsmB[t].q[0];

      if (t>0)
         add_qflt(rqsmB[t].q,rqsmB[0].q,rqsmB[0].q);
   }

   if ((bc==1)||(bc==2))
      add_qflt(rqsmE[N0-1].q,rqsmB[0].q,rqsmB[0].q);

   return rqsmB[0].q[0];
}
