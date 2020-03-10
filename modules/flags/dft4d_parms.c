
/*******************************************************************************
*
* File dft4d_parms.c
*
* Copyright (C) 2015, 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* DFT4D parameter data base.
*
*   int set_dft4d_parms(int *idp,int *nx,int csize)
*     Initializes a new instance of a DFT4D parameter set for the Fourier
*     transformation of fields in four dimensions (see the notes). The
*     program returns the id of the parameter set in the data base.
*
*   void reset_dft4d_parms(int id,int csize)
*     Resets the parameter csize in the DFT4D parameter set with the given
*     id. An error occurs if there is no such parameter set.
*
*   dft4d_parms_t *dft4d_parms(int id)
*     Returns the DFT4D parameter set with the given id. If there is no
*     such parameter set, the program returns NULL.
*
* The discrete Fourier transform operates on arrays of double-precision
* complex numbers. Different types of Fourier transforms are implemented
* as described in the notes
*
*  M. Luescher: "Discrete Fourier transform", January 2015, doc/dft.pdf,
*
* and on the top of the module dft/fft.c.
*
* The parameters administered by this module specify the Fourier transform
* of fields in four dimensions. In practice the fields will live on the
* lattice defined by the macros in global.h, but the programs that perform
* the 4d Fourier transforms do not assume this. Instead the fields are
* expected in the form of local complex_dble arrays
*
*  f[s+csize*ix],
*
*  0<=s<csize,
*
*  ix=x3+nx[3]*x2+nx[2]*nx[3]*x1+nx[1]*nx[2]*nx[3]*x0,
*
*  0<=x0<nx[0], 0<=x1<nx[1], 0<=x2<nx[2], 0<=x3<nx[3].
*
* Here s and csize allow fields with internal indices (with range csize) to
* be accommodated, while x0,x1,x2,x3 and nx[0],nx[1],nx[2],nx[3] are the
* coordinates and sizes of the local lattice on which the field f lives.
* Together with the globally defined Cartesian process grid, these local
* lattices define a 4d global lattice in the obvious way.
*
* Apart from csize and the array nx, the program set_dft4d_parms() has only
* one further argument, idp, which must be an array with the 4 elements
*
*  idp[mu]    Id of the parameter set specifying the Fourier transform
*             in direction mu=0,..,3 (see dft/dft_parms.c).
*
* The program internally calculates the arrays
*
*  ny[mu]     = vol/nx[mu] where vol=nx[0]*nx[1]*nx[2]*nx[3],
*
*  nf[mu]     = csize*ny[mu],
*
*  nx[mu][i]  = values of nx[mu] at cpr[mu]=i and fixed cpr[nu!=mu]
*               (i=0,1,..,nproc[mu]-1),
*
*  mf[mu][i]  = m+(i<r), where m=nf[mu]/nproc[mu] and r=nf[mu]%nproc[mu]
*               (i=0,1,..,nproc[mu]-1),
*
*  dp[mu]     = dft_parms(idp[mu]),
*
* where nproc={NPROC0,NPROC1,NPROC2,NPROC3}. Together with csize, all these
* arrays are returned by dft4d_parms() in a dft4d_parms_t structure.
*
* The values of csize and idp[mu] are required to be globally the same. The
* local lattice sizes, nx[mu], may however depend on the process id subject
* to the following constraints:
*
*  - The lattice sizes nx[mu] must all be greater or equal to 1.
*
*  - nx[mu] may depend on cpr[mu] but must be constant at fixed cpr[mu]
*    and varying cpr[nu!=mu].
*
*  - The sum nx[mu][i] over i from 0 to nproc[mu]-1 must be equal to
*    dp[mu].n if dp[mu].type=EXP and otherwise equal to dp[mu].n+1.
*
* An error occurs if these constraints are violated.
*
* The program set_dft4d_parms() performs global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define DFT4D_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"
#include "utils.h"
#include "lattice.h"
#include "flags.h"
#include "global.h"

static const int nproc[4]={NPROC0,NPROC1,NPROC2,NPROC3};
static int mxs=0,ids=0;
static dft4d_parms_t **dpa;


static void alloc_dpa(void)
{
   int i,mu,*na;
   dft4d_parms_t **dd,*d;

   error(mxs>(INT_MAX-16),1,"alloc_dpa [dft4d_parms.c]",
         "Unable to allocate dft4d_parms_t structures");

   na=malloc(32*(nproc[0]+nproc[1]+nproc[2]+nproc[3])*sizeof(*na));
   dd=malloc((mxs+16)*sizeof(*dd));
   d=malloc(16*sizeof(*d));
   error((na==NULL)||(dd==NULL)||(d==NULL),2,"alloc_dpa [dft4d_parms.c]",
         "Unable to allocate dft4d_parms_t structures");

   for (i=0;i<mxs;i++)
      dd[i]=dpa[i];

   for (i=0;i<16;i++)
   {
      d[i].csize=0;

      for (mu=0;mu<4;mu++)
      {
         d[i].nx[mu]=na;
         na+=nproc[mu];
         d[i].mf[mu]=na;
         na+=nproc[mu];
      }

      dd[mxs+i]=d+i;
   }

   if (mxs>0)
      free(dpa);

   dpa=dd;
   mxs+=16;
}


static void set_nx(dft4d_parms_t *dp)
{
   int i,mu,nu,dmy,snx,*nx;
   int np,ip0,ip1,ie,n[4],nx0[3],nx1[3];
   MPI_Status stat;

   n[0]=cpr[0];
   n[1]=cpr[1];
   n[2]=cpr[2];
   n[3]=cpr[3];
   dmy=0;
   ie=0;

   for (mu=0;mu<4;mu++)
   {
      i=0;

      for (nu=0;nu<4;nu++)
      {
         if (nu!=mu)
         {
            nx0[i]=(*dp).nx[nu][cpr[nu]];
            nx1[i]=nx0[i];
            i+=1;
         }
      }

      nx=(*dp).nx[mu];
      np=nproc[mu];

      if (np>1)
      {
         n[mu]=0;
         ip0=ipr_global(n);

         for (i=1;i<np;i++)
         {
            n[mu]=i;
            ip1=ipr_global(n);

            if (cpr[mu]==0)
            {
               MPI_Send(nx0,3,MPI_INT,ip1,0,MPI_COMM_WORLD);
               MPI_Recv(nx+i,1,MPI_INT,ip1,i,MPI_COMM_WORLD,&stat);
            }
            else if (cpr[mu]==i)
            {
               MPI_Recv(nx1,3,MPI_INT,ip0,0,MPI_COMM_WORLD,&stat);
               MPI_Send(nx+i,1,MPI_INT,ip0,i,MPI_COMM_WORLD);
            }
         }

         for (i=1;i<np;i++)
         {
            n[mu]=i;
            ip1=ipr_global(n);

            if (cpr[mu]==0)
            {
               MPI_Send(nx,np,MPI_INT,ip1,0,MPI_COMM_WORLD);
               MPI_Recv(&dmy,1,MPI_INT,ip1,i,MPI_COMM_WORLD,&stat);
            }
            else if (cpr[mu]==i)
            {
               MPI_Recv(nx,np,MPI_INT,ip0,0,MPI_COMM_WORLD,&stat);
               MPI_Send(&dmy,1,MPI_INT,ip0,i,MPI_COMM_WORLD);
            }
         }

         n[mu]=cpr[mu];
      }

      if ((*dp).dp[mu][0].type==EXP)
         snx=nx[0];
      else
         snx=nx[0]-1;

      for (i=1;i<np;i++)
         snx+=nx[i];

      for (i=0;i<3;i++)
         ie|=(nx0[i]!=nx1[i]);

      error((snx!=(*dp).dp[mu][0].n)||(ie!=0),1,
         "set_nx [dft4d_parms.c]","Inconsistent parameter choice");
   }
}


int set_dft4d_parms(int *idp,int *nx,int csize)
{
   int id,iprms[5];
   int mu,vol,np,ny,nf;
   int m,r,i;
   dft4d_parms_t *dp;

   error_root(iup[0][0]==0,1,"set_dft4d_parms [dft4d_parms.c]",
              "Geometry arrays are not set");

   error((nx[0]<1)||(nx[1]<1)||(nx[2]<1)||(nx[3]<1)||(csize<1),1,
         "set_dft4d_parms [dft4d_parms.c]","nx or csize is out of range");

   if (NPROC>1)
   {
      iprms[0]=idp[0];
      iprms[1]=idp[1];
      iprms[2]=idp[2];
      iprms[3]=idp[3];
      iprms[4]=csize;

      MPI_Bcast(iprms,5,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=idp[0])||(iprms[1]!=idp[1])||(iprms[2]!=idp[2])||
            (iprms[3]!=idp[3])||(iprms[4]!=csize),1,
            "set_dft4d_parms [dft4d_parms.c]","Parameters are not global");
   }

   if (ids==mxs)
      alloc_dpa();

   id=ids;
   ids+=1;
   dp=dpa[id];

   (*dp).csize=csize;
   vol=nx[0]*nx[1]*nx[2]*nx[3];

   for (mu=0;mu<4;mu++)
   {
      np=nproc[mu];
      ny=vol/nx[mu];
      nf=csize*ny;
      m=nf/np;
      r=nf%np;

      (*dp).ny[mu]=ny;
      (*dp).nf[mu]=nf;
      (*dp).nx[mu][cpr[mu]]=nx[mu];

      for (i=0;i<np;i++)
         (*dp).mf[mu][i]=m+(i<r);

      (*dp).dp[mu]=dft_parms(idp[mu]);
   }

   error_root(((*dp).dp[0]==NULL)||((*dp).dp[1]==NULL)||
              ((*dp).dp[2]==NULL)||((*dp).dp[3]==NULL),1,
              "set_dft4d_parms [dft4d_parms.c]","DFT parameters are not set");
   set_nx(dp);

   return id;
}


void reset_dft4d_parms(int id,int csize)
{
   int mu,vol,np,ny,nf;
   int m,r,i;
   int nx[4],iprms[2];
   dft4d_parms_t *dp;

   error_root((id<0)||(id>=ids)||(csize<1),1,
              "reset_dft4d_parms [dft4d_parms.c]",
              "Parameters are out of range");

   if (NPROC>1)
   {
      iprms[0]=id;
      iprms[1]=csize;

      MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=id)||(iprms[1]!=csize),1,
            "reset_dft4d_parms [dft4d_parms.c]","Parameters are not global");
   }

   dp=dpa[id];

   if (csize!=(*dp).csize)
   {
      (*dp).csize=csize;

      for (mu=0;mu<4;mu++)
         nx[mu]=(*dp).nx[mu][cpr[mu]];

      vol=nx[0]*nx[1]*nx[2]*nx[3];

      for (mu=0;mu<4;mu++)
      {
         ny=vol/nx[mu];
         nf=csize*ny;
         (*dp).nf[mu]=nf;

         np=nproc[mu];
         m=nf/np;
         r=nf%np;

         for (i=0;i<np;i++)
            (*dp).mf[mu][i]=m+(i<r);
      }
   }
}


dft4d_parms_t *dft4d_parms(int id)
{
   if ((id>=0)&&(id<ids))
      return dpa[id];
   else
      return NULL;
}
