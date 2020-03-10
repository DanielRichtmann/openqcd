
/*******************************************************************************
*
* File dpgen.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generation of random DFT4D parameter sets.
*
*   int dpgen(void)
*     Returns the id of a newly generated random DFT4D parameter set
*     (see pmdb/dft4d_parms.c).
*
*   void print_dft4d_parms(int id)
*     Prints the DFT4D parameters with the specified id to stdout from
*     process 0.
*
* The program dpgen() acts globally and must be called on all MPI processes
* simultaneously.
*
*******************************************************************************/

#define DPGEN_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "random.h"
#include "flags.h"
#include "global.h"

static int nproc[4]={NPROC0,NPROC1,NPROC2,NPROC3};
static int pm[4],idp[4],nx[4];


static void set_idp(void)
{
   int my_rank,mu,n;
   double r;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   for (mu=0;mu<4;mu++)
   {
      if (my_rank==0)
      {
         ranlxd(&r,1);
         pm[0]=(int)((double)(DFT_TYPES)*r);

         ranlxd(&r,1);

         for (n=(int)(3.0*r)+3;n<nproc[mu];n*=2);

         ranlxd(&r,1);

         if (r<0.5)
            pm[1]=2*n;
         else
            pm[1]=4*n;

         ranlxd(&r,1);

         if (r<0.5)
            pm[2]=0;
         else
            pm[2]=1;

         ranlxd(&r,1);

         if (r<0.5)
            pm[3]=0;
         else
            pm[3]=1;
      }

      MPI_Bcast(pm,4,MPI_INT,0,MPI_COMM_WORLD);
      idp[mu]=set_dft_parms((dft_type_t)(pm[0]),pm[1],pm[2],pm[3]);
   }
}


static void set_nx(void)
{
   int mu,n,m,r;
   dft_parms_t *dp;

   for (mu=0;mu<4;mu++)
   {
      dp=dft_parms(idp[mu]);
      n=(*dp).n;

      if ((*dp).type!=EXP)
         n+=1;

      m=n/nproc[mu];
      r=n%nproc[mu];

      nx[mu]=m+(cpr[mu]<r);
   }
}


int dpgen(void)
{
   double r;

   set_idp();
   set_nx();
   ranlxd(&r,1);
   MPI_Bcast(&r,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return set_dft4d_parms(idp,nx,(int)(8.0*r)+1);
}


void print_dft4d_parms(int id)
{
   int my_rank,mu,i;
   dft_parms_t *dp;
   dft4d_parms_t *dp4d;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("DFT4D parameter set with id = %d\n",id);
      dp4d=dft4d_parms(id);
      error_root(dp4d==NULL,1,"print_dft4d_parms [dpgen.c]",
                 "Parameter set is not defined");
      printf("csize = %d\n",(*dp4d).csize);

      for (mu=0;mu<4;mu++)
      {
         dp=(*dp4d).dp[mu];

         printf("mu = %d, ",mu);

         if ((*dp).type==EXP)
            printf("type = EXP\n");
         else if ((*dp).type==COS)
            printf("type = COS\n");
         else if ((*dp).type==SIN)
            printf("type = SIN\n");
         else
            error_root(1,1,"print_dft4d_parms [dpgen.c]",
                       "Unknown type of Fourier transform");

         printf("n = %d, local sizes nx = %d",(*dp).n,(*dp4d).nx[mu][0]);

         for (i=1;i<nproc[mu];i++)
            printf(",%d",(*dp4d).nx[mu][i]);

         printf("\n");
         printf("b = %d, c = %d\n",(*dp).b,(*dp).c);
      }
   }
}
