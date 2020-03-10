
/*******************************************************************************
*
* File pt2pt.c
*
* Copyright (C) 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of point-to-point correlation functions.
*
*   void pt2pt(int i3d,int r,double *f,
*              complex_dble *rf,double *w1,double *w2,double *obs)
*     Constructs an observable obs whose average is the point-to-point
*     correlation function of the observable field f at distance r (see
*     the notes). The fields rf, w1 and w2 are used as workspace and
*     the average over directions is taken in 3d (i3d=1) or 4d (i3d!=1).
*
* The programs in this module assume periodic boundary conditions in all
* directions. An error occurs if this is not the case. The program acts
* on global real scalar fields f[ix], where 0<=ix<VOLUME is the index of
* the points on the local lattice defined by the geometry routines (see
* main/README.global).
*
* The observable constructed by pt2pt() is given by
*
*   obs(x)=(1/4)*sum_{mu=0}^3{f(x+hp*e_mu)*f(x-hm*e_mu)}
*
*   hp=r-hm, hm=r/2, e_mu: unit vector in direction mu.
*
* If i3d=1 the average over the direction mu is restricted to the space
* directions mu=1,2,3.
*
* Usually its average value is removed from the field f(x) before calling
* pt2pt() so that the average of obs(x) yields a stochastic estimate of the
* connected correlation function. See section 3.3 in
*
*   M. Lüscher,
*   "Stochastic locality and master-field simulations of very large lattices",
*   EPJ Web Conf. 175 (2018) 01002,
*
* for further explanations.
*
* The program in this module performs global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define PT2PT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "msfcts.h"
#include "global.h"


void pt2pt(int i3d,int r,double *f,
           complex_dble *rf,double *w1,double *w2,double *obs)
{
   int mu,s[4],iprms[2];

   if (NPROC>1)
   {
      iprms[0]=i3d;
      iprms[1]=r;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=i3d)||(iprms[1]!=r),1,"pt2pt [pt2pt.c]",
            "Parameters are not global");
   }

   i3d=(i3d==1);
   s[0]=0;
   s[1]=0;
   s[2]=0;
   s[3]=0;

   for (mu=i3d;mu<4;mu++)
   {
      s[mu]=r-(r/2);
      shift_fld(s,f,rf,w1);
      s[mu]-=r;

      if (mu==i3d)
      {
         shift_fld(s,f,rf,obs);
         mul_flds(obs,w1);
      }
      else
      {
         shift_fld(s,f,rf,w2);
         mul_flds(w2,w1);
         add_flds(obs,w2);
      }

      s[mu]=0;
   }

   if (i3d)
      mulr_fld(1.0/3.0,obs);
   else
      mulr_fld(1.0/4.0,obs);
}
