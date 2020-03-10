
/*******************************************************************************
*
* File dft4d.c
*
* Copyright (C) 2015 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Parallel 4d Fourier transform.
*
*   void dft4d(int id,complex_dble *f,complex_dble *ft)
*     Applies the Fourier transform specified by the DFT4D parameter set
*     number id to the function f and assigns the calculated function to
*     ft (see the notes).
*
*   void inv_dft4d(int id,complex_dble *ft,complex_dble *f)
*     Applies the inverse of the Fourier transform specified by the DFT4D
*     parameter set number id to the function ft and assigns the calculated
*     function to f (see the notes).
*
* The programs in this module expect the function f and ft to be given in
* the form specified by the parameter set number id, as described at the top
* of the module dft4d_parms.c. On exit the input array is unchanged unless
* it overlaps with the output array (which is permissible).
*
*******************************************************************************/

#define DFT4D_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "dft.h"
#include "global.h"

static int ngs=0,nga=0;
static int n[4],*ny,*nf,**nx,**mf;
static complex_dble **gs,**gts,**hs,**hts;
static complex_dble *ga,*fs,*fts;


static void alloc_arrays(void)
{
   int mu,nm,nfm,nxm,mfm,m,i;

   nm=0;
   nfm=0;
   nxm=0;
   mfm=0;

   for (mu=0;mu<4;mu++)
   {
      i=cpr[mu];

      if (n[mu]>nm)
         nm=n[mu];
      if (nf[mu]>nfm)
         nfm=nf[mu];
      if (nx[mu][i]>nxm)
         nxm=nx[mu][i];
      if (mf[mu][i]>mfm)
         mfm=mf[mu][i];
   }

   m=2*(nxm+nm);

   if (ngs<m)
   {
      if (ngs>0)
         free(gs);

      gs=malloc(m*sizeof(*gs));
      ngs=m;
   }

   m=nxm*nfm+2*nm*mfm;

   if (nga<m)
   {
      if (nga>0)
         afree(ga);

      ga=amalloc(m*sizeof(*ga),ALIGN);
      nga=m;
   }

   error((gs==NULL)||(ga==NULL),1,"alloc_arrays [dft4d.c]",
         "Unable to allocate auxiliary field arrays");
}


static void set_arrays(int mu)
{
   int i,ni,nfi,nxi,mfi;
   complex_dble *g0,*g1,*h0,*h1;

   ni=n[mu];
   nfi=nf[mu];
   nxi=nx[mu][cpr[mu]];
   mfi=mf[mu][cpr[mu]];

   gts=gs+nxi;
   hs=gts+nxi;
   hts=hs+ni;

   if (mu==0)
      g0=fs;
   else
      g0=fts;

   g1=ga;

   for (i=0;i<nxi;i++)
   {
      gs[i]=g0;
      g0+=nfi;
      gts[i]=g1;
      g1+=nfi;
   }

   h0=g1;
   h1=h0+ni*mfi;

   for (i=0;i<ni;i++)
   {
      hs[i]=h0;
      h0+=mfi;
      hts[i]=h1;
      h1+=mfi;
   }
}


void dft4d(int id,complex_dble *f,complex_dble *ft)
{
   int mu,csize,iprms[1];
   dft_parms_t **dfp;
   dft4d_parms_t *dp;

   dp=dft4d_parms(id);
   error_root(dp==NULL,1,"dft4d [dft4d.c]",
              "Unknown parameter set");

   if (NPROC>1)
   {
      iprms[0]=id;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=id,1,"dft4d [dft4d.c]",
            "Parameter set is not global");
   }

   ny=(*dp).ny;
   nf=(*dp).nf;
   nx=(*dp).nx;
   mf=(*dp).mf;
   csize=(*dp).csize;
   dfp=(*dp).dp;

   for (mu=0;mu<4;mu++)
      n[mu]=dfp[mu][0].n+(dfp[mu][0].type!=EXP);

   alloc_arrays();
   fs=f;
   fts=ft;

   for (mu=0;mu<4;mu++)
   {
      set_arrays(mu);
      dft_gather(mu,nx[mu],mf[mu],gs,hs);
      fft(dfp[mu],mf[mu][cpr[mu]],hs,hts);
      dft_scatter(mu,nx[mu],mf[mu],hts,gts);
      dft_shuf(nx[mu][cpr[mu]],ny[mu],csize,ga,fts);
   }
}


void inv_dft4d(int id,complex_dble *ft,complex_dble *f)
{
   int mu,csize,iprms[1];
   dft_parms_t **dfp;
   dft4d_parms_t *dp;

   dp=dft4d_parms(id);
   error_root(dp==NULL,1,"inv_dft4d [dft4d.c]",
              "Unknown parameter set");

   if (NPROC>1)
   {
      iprms[0]=id;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=id,1,"inv_dft4d [dft4d.c]",
            "Parameter set is not global");
   }

   ny=(*dp).ny;
   nf=(*dp).nf;
   nx=(*dp).nx;
   mf=(*dp).mf;
   csize=(*dp).csize;
   dfp=(*dp).dp;

   for (mu=0;mu<4;mu++)
      n[mu]=dfp[mu][0].n+(dfp[mu][0].type!=EXP);

   alloc_arrays();
   fs=ft;
   fts=f;

   for (mu=0;mu<4;mu++)
   {
      set_arrays(mu);
      dft_gather(mu,nx[mu],mf[mu],gs,hs);
      inv_fft(dfp[mu],mf[mu][cpr[mu]],hs,hts);
      dft_scatter(mu,nx[mu],mf[mu],hts,gts);
      dft_shuf(nx[mu][cpr[mu]],ny[mu],csize,ga,fts);
   }
}
