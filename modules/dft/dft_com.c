
/*******************************************************************************
*
* File dft_com.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication programs for the parallel discrete Fourier transform.
*
*   void dft_gather(int mu,int *nx,int *mf,complex_dble **lf,complex_dble **f)
*     Gathers the locally stored parts lf of a set of nf functions from the
*     MPI processes in direction mu so as to reconstruct a subsequence of
*     the full functions on each process (see the notes).
*
*   void dft_scatter(int mu,int *nx,int *mf,complex_dble **f,complex_dble **lf)
*     Performs the inverse of the operation performed by dft_gather().
*
* These programs operate on sets of functions of the space coordinate x in
* the specified direction mu. The functions are stored on the computer in
* one of two ways:
*
*  1. On each process, there is a subsequence of functions f[x][i], where
*     i labels the functions and x runs over the full range of the lattice
*     coordinate in direction mu. While the range of x must be the same
*     on all MPI processes, the number of functions stored may depend on
*     the process id.
*
*  2. The same functions may be split among the MPI processes in direction
*     mu and be locally represented by functions lf[x][i]. In this case, x
*     runs over an interval on the coordinate axis in direction mu and the
*     range of the index 0<=i<nf includes all functions. The x intervals are
*     non-overlapping and together cover the full range of the coordinate.
*
* The functions dft_gather() and dft_scatter() copy the functions from one
* representation to the other. In both cases, the parameters have the same
* meaning:
*
*  mu            Lattice direction in which the functions are split up.
*                Independently of the storage format, the functions are
*                stored on the nproc[mu] MPI processes in direction mu with
*                fixed coordinates cpr[nu!=mu] (on the full machine, there
*                are thus NPROC/nproc[mu] independent sets of functions).
*
*  nx[k]         Range of the coordinate x on the process with cpr[mu]=k
*                and the same cpr[nu!=mu] as the local process. nx need
*                not be the same on all processes but must be positive.
*                The full range of the coordinate is 0<=x<{sum_k nx[k]}.
*
*  mf[k]         Number of the functions f[x][i] on the process with
*                cpr[mu]=k. The number is m+1 if k<r and m if k>=r,
*                where m=nf/nproc[mu] and r=nf%nproc[mu].
*
*  f             Array of double-precision complex numbers representing
*                a subset of the functions in the storage format 1. The
*                array size must be {sum_j nx[j]} x mf[k] on the process
*                with cpr[mu]=k.
*
*  lf            Array of double-precision complex numbers representing
*                the functions in storage format 2. The array size must
*                be nx[k] x {sum_j mf[j]} on the process with cpr[mu]=k.
*
* The program dft_gather() moves the data among the processes at fixed
* cpr[nu!=mu] such that the array f on the process with cpr[mu]=0 contains
* the first m (or m+1) functions, on the process with cpr[mu]=1 the next
* m (or m+1) functions, and so on, all given over the full range of the
* space coordinate x.
*
* While the process grid is set by the macros in global.h, no relation is
* assumed between the lattice sizes defined there and the values of the
* parameter nx.
*
* It is up to the user to ensure the correct sizes of the field arrays
* and their consistency with the size arrays nx and mf.
*
*******************************************************************************/

#define DFT_COM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "dft.h"
#include "global.h"

typedef struct
{
   int np,is,*ipa;
   int *nx,*snx;
   int *mf,*smf;
} fcom_t;

static const int nproc[4]={NPROC0,NPROC1,NPROC2,NPROC3};
static fcom_t fcom[4]={{0,0,NULL,NULL,NULL,NULL,NULL},
                       {0,0,NULL,NULL,NULL,NULL,NULL},
                       {0,0,NULL,NULL,NULL,NULL,NULL},
                       {0,0,NULL,NULL,NULL,NULL,NULL}};


static void init_fcom(int mu)
{
   int i,np,is,n[4];
   int *ipa;

   np=nproc[mu];
   is=(cpr[0]+cpr[1]+cpr[2]+cpr[3]-cpr[mu])&0x1;

   ipa=malloc(3*np*sizeof(*ipa));
   error(ipa==NULL,1,"init_fcom [dft_com.c]",
         "Unable to allocate fcom_t data arrays");

   n[0]=cpr[0];
   n[1]=cpr[1];
   n[2]=cpr[2];
   n[3]=cpr[3];

   for (i=0;i<np;i++)
   {
      n[mu]=i;
      ipa[i]=ipr_global(n);
   }

   fcom[mu].np=np;
   fcom[mu].is=is;
   fcom[mu].ipa=ipa;
   fcom[mu].snx=ipa+np;
   fcom[mu].smf=ipa+2*np;
}


static void set_fcom(int mu,int *nx,int *mf)
{
   int i,np,*snx,*smf;

   if (fcom[mu].np==0)
      init_fcom(mu);

   np=fcom[mu].np;
   snx=fcom[mu].snx;
   smf=fcom[mu].smf;

   fcom[mu].nx=nx;
   fcom[mu].mf=mf;

   snx[0]=0;
   smf[0]=0;

   for (i=1;i<np;i++)
   {
      snx[i]=snx[i-1]+nx[i-1];
      smf[i]=smf[i-1]+mf[i-1];
   }
}


static int round_robin(int np,int r,int cp0)
{
   int cp1;

   if (cp0==0)
      cp1=np-1-r;
   else
   {
      cp1=np-1-cp0-r;
      if (cp1<0)
         cp1+=(np-1);

      if (cp1>0)
      {
         cp1-=r;
         if (cp1<=0)
            cp1+=(np-1);
      }
   }

   return cp1;
}


static void gather_fcts(int mu,complex_dble **lf,complex_dble **f)
{
   int np,is,*ipa;
   int *nx,*snx;
   int *mf,*smf;
   int i,x,r,cp0,cp1,ip1;
   complex_dble *w1,*w2;
   MPI_Status stat;

   np=fcom[mu].np;
   is=fcom[mu].is;
   ipa=fcom[mu].ipa;
   nx=fcom[mu].nx;
   snx=fcom[mu].snx;
   mf=fcom[mu].mf;
   smf=fcom[mu].smf;
   cp0=cpr[mu];

   for (x=0;x<nx[cp0];x++)
   {
      w1=f[x+snx[cp0]];
      w2=lf[x]+smf[cp0];

      for (i=0;i<mf[cp0];i++)
         w1[i]=w2[i];
   }

   for (r=0;r<(np-1);r++)
   {
      cp1=round_robin(np,r,cp0);
      ip1=ipa[cp1];

      for (x=0;(x<nx[cp0])||(x<nx[cp1]);x++)
      {
         if ((cp0<cp1)^is)
         {
            if ((x<nx[cp0])&&(mf[cp1]>0))
               MPI_Send(lf[x]+smf[cp1],2*mf[cp1],MPI_DOUBLE,ip1,
                        x+snx[cp0],MPI_COMM_WORLD);
            if ((x<nx[cp1])&&(mf[cp0]>0))
               MPI_Recv(f[x+snx[cp1]],2*mf[cp0],MPI_DOUBLE,ip1,
                        x+snx[cp1],MPI_COMM_WORLD,&stat);
         }
         else
         {
            if ((x<nx[cp1])&&(mf[cp0]>0))
               MPI_Recv(f[x+snx[cp1]],2*mf[cp0],MPI_DOUBLE,ip1,
                        x+snx[cp1],MPI_COMM_WORLD,&stat);
            if ((x<nx[cp0])&&(mf[cp1]>0))
               MPI_Send(lf[x]+smf[cp1],2*mf[cp1],MPI_DOUBLE,ip1,
                        x+snx[cp0],MPI_COMM_WORLD);
         }
      }
   }
}


void dft_gather(int mu,int *nx,int *mf,complex_dble **lf,complex_dble **f)
{
   int iprms[1];

   error_root((mu<0)||(mu>3),1,"dft_gather [dft_com.c]",
              "Parameter mu is out of range");

   if (NPROC>1)
   {
      iprms[0]=mu;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=mu,1,"dft_gather [dft_com.c]",
            "Parameter mu is not global");
   }

   set_fcom(mu,nx,mf);
   gather_fcts(mu,lf,f);
}


static void scatter_fcts(int mu,complex_dble **f,complex_dble **lf)
{
   int np,is,*ipa;
   int *nx,*snx;
   int *mf,*smf;
   int i,x,r,cp0,cp1,ip1;
   complex_dble *w1,*w2;
   MPI_Status stat;

   np=fcom[mu].np;
   is=fcom[mu].is;
   ipa=fcom[mu].ipa;
   nx=fcom[mu].nx;
   snx=fcom[mu].snx;
   mf=fcom[mu].mf;
   smf=fcom[mu].smf;
   cp0=cpr[mu];

   for (x=0;x<nx[cp0];x++)
   {
      w1=lf[x]+smf[cp0];
      w2=f[x+snx[cp0]];

      for (i=0;i<mf[cp0];i++)
         w1[i]=w2[i];
   }

   for (r=0;r<(np-1);r++)
   {
      cp1=round_robin(np,r,cp0);
      ip1=ipa[cp1];

      for (x=0;(x<nx[cp0])||(x<nx[cp1]);x++)
      {
         if ((cp0<cp1)^is)
         {
            if ((x<nx[cp0])&&(mf[cp1]>0))
               MPI_Recv(lf[x]+smf[cp1],2*mf[cp1],MPI_DOUBLE,ip1,
                        x+snx[cp0],MPI_COMM_WORLD,&stat);
            if ((x<nx[cp1])&&(mf[cp0]>0))
               MPI_Send(f[x+snx[cp1]],2*mf[cp0],MPI_DOUBLE,ip1,
                        x+snx[cp1],MPI_COMM_WORLD);
         }
         else
         {
            if ((x<nx[cp1])&&(mf[cp0]>0))
               MPI_Send(f[x+snx[cp1]],2*mf[cp0],MPI_DOUBLE,ip1,
                        x+snx[cp1],MPI_COMM_WORLD);
            if ((x<nx[cp0])&&(mf[cp1]>0))
               MPI_Recv(lf[x]+smf[cp1],2*mf[cp1],MPI_DOUBLE,ip1,
                        x+snx[cp0],MPI_COMM_WORLD,&stat);
         }
      }
   }
}


void dft_scatter(int mu,int *nx,int *mf,complex_dble **f,complex_dble **lf)
{
   int iprms[1];

   error_root((mu<0)||(mu>3),1,"dft_scatter [dft_com.c]",
              "Parameter mu is out of range");

   if (NPROC>1)
   {
      iprms[0]=mu;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=mu,1,"dft_scatter [dft_com.c]",
            "Parameter mu is not global");
   }

   set_fcom(mu,nx,mf);
   scatter_fcts(mu,f,lf);
}
