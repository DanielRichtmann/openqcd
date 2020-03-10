
/*******************************************************************************
*
* File wspace.c
*
* Copyright (C) 2010-2013, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Workspace administration.
*
*   void alloc_wud(int n)
*     Allocates a workspace of n double-precision gauge fields.
*
*   su3_dble **reserve_wud(int n)
*     Reserves a new workspace of n global double-precision gauge fields
*     and returns the array ud[0],..,ud[n-1] of the base addresses of the
*     fields in the workspace. No workspace is reserved and a NULL pointer
*     is returned if n<=0.
*
*   int release_wud(void)
*     Releases the workspace of global double-precision gauge fields that
*     was last reserved and returns the number of fields that are released.
*
*   void alloc_wfd(int n)
*     Allocates a workspace of n double-precision force fields.
*
*   su3_alg_dble **reserve_wfd(int n)
*     Reserves a new workspace of n global double-precision force fields
*     and returns the array fd[0],..,fd[n-1] of the base addresses of the
*     fields in the workspace. No workspace is reserved and a NULL pointer
*     is returned if n<=0.
*
*   int release_wfd(void)
*     Releases the workspace of global double-precision force fields that
*     was last reserved and returns the number of fields that are released.
*
*   void alloc_ws(int n)
*     Allocates a workspace of n single-precision spinor fields.
*
*   spinor **reserve_ws(int n)
*     Reserves a new workspace of n global single-precision spinor fields
*     and returns the array s[0],..,s[n-1] of the base addresses of the
*     fields in the workspace. No workspace is reserved and a NULL pointer
*     is returned if n<=0.
*
*   int release_ws(void)
*     Releases the workspace of global single-precision spinor fields that
*     was last reserved and returns the number of fields that are released.
*
*   void alloc_wsd(int n)
*     Allocates a workspace of n double-precision spinor fields.
*
*   void wsd_uses_ws(void)
*     This program may be called in place of alloc_wsd() if the workspace
*     for the global single-precision spinor fields is to be used for the
*     double-precision fields too. If a workspace of n single-precision
*     fields is allocated, nsd double- and ns single-precision fields can
*     then be accommodated as long as 2*nsd+ns<=n.
*
*   spinor_dble **reserve_wsd(int n)
*     Reserves a new workspace of n global double-precision spinor fields
*     and returns the array sd[0],..,sd[n-1] of the base addresses of the
*     fields in the workspace. No workspace is reserved and a NULL pointer
*     is returned if n<=0.
*
*   int release_wsd(void)
*     Releases the workspace of global double-precision spinor fields that
*     was last reserved and returns the number of fields that are released.
*
*   void alloc_wv(int n)
*     Allocates a workspace of n single-precision vector fields.
*
*   complex **reserve_wv(int n)
*     Reserves a new workspace of n global single-precision vector fields
*     and returns the array v[0],..,v[n-1] of the base addresses of the
*     fields in the workspace. No workspace is reserved and a NULL pointer
*     is returned if n<=0.
*
*   int release_wv(void)
*     Releases the workspace of global single-precision vector fields that
*     was last reserved and returns the number of fields that are released.
*
*   void alloc_wvd(int n)
*     Allocates a workspace of n double-precision vector fields.
*
*   complex_dble **reserve_wvd(int n)
*     Reserves a new workspace of n global double-precision vector fields
*     and returns the array vd[0],..,vd[n-1] of the base addresses of the
*     fields in the workspace. No workspace is reserved and a NULL pointer
*     is returned if n<=0.
*
*   int release_wvd(void)
*     Releases the workspace of global double-precision vector fields that
*     was last reserved and returns the number of fields that are released.
*
*   size_t wsp_msize(void)
*     Returns the total size in bytes of the locally allocated workspaces.
*
*   void print_wsp(void)
*     Prints the numbers of fields in the workspaces and the relative size
*     of the memory used by the workspaces to stdout on MPI process 0. The
*     program does nothing if the workspace is empty.
*
* By definition a workspace is a set of global fields which is reserved for
* the current program. This module allows to assign and release the required
* workspaces dynamically in such a way that the privacy of the workspaces is
* guaranteed.
*
* The total workspace is the set of all allocated fields of a given type. All
* programs must reserve the required workspaces and must release them before
* the program returns to the calling program. It is possible to reserve several
* workspaces of the same type in a given program. A corresponding number of
* calls to w*_release() is then required to release all the workspaces that
* were reserved.
*
* The fields in the gauge and force workspaces have 4*VOLUME elements and can
* only store a copy of the field variables on the local lattice. In the case
* of the spinor and spinor_dble fields, the number of spinors allocated per
* field is NSPIN (see include/global.h). Vector fields are arrays of complex
* numbers on which the little Dirac operator acts. They have Ns*(nb+nbb/2)
* elements, where Ns is the number of local deflation modes, while nb and nbb
* are the numbers blocks in the DFL_BLOCKS grid and its exterior boundary.
*
* Workspaces can be freed or reallocated by calling alloc_w*() if no fields
* in the workspace are currently in use. In particular, a workspace can be
* freed by calling alloc_w*(0).
*
* If single- and double-precision spinor fields share the same workspace, the
* release_ws() and release_wsd() statements must appear in the inverse order
* of the corresponding reserve_ws() and reserve_wsd() statements. An error
* occurs if this is not the case.
*
* The programs in this module may perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define WSPACE_C

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "global.h"

typedef struct
{
   int type;
   int n,nres,ires,*nw;
   void **w0,**w;
} wspace_t;

static int *iws=NULL,usews=0;
static wspace_t wud={0,0,0,0,NULL,NULL,NULL};
static wspace_t wfd={1,0,0,0,NULL,NULL,NULL};
static wspace_t ws= {2,0,0,0,NULL,NULL,NULL};
static wspace_t wsd={3,0,0,0,NULL,NULL,NULL};
static wspace_t wv= {4,0,0,0,NULL,NULL,NULL};
static wspace_t wvd={5,0,0,0,NULL,NULL,NULL};


static size_t fld_size(int type)
{
   int nvec,*bs;
   dfl_parms_t dfl;

   if (type==0)
      return sizeof(su3_dble)*(size_t)(4*VOLUME);
   else if (type==1)
      return sizeof(su3_alg_dble)*(size_t)(4*VOLUME);
   else if (type==2)
      return sizeof(spinor)*(size_t)(NSPIN);
   else if (type==3)
      return sizeof(spinor_dble)*(size_t)(NSPIN);
   else
   {
      dfl=dfl_parms();
      error(dfl.Ns==0,1,"fld_size [wspace.c]",
            "The deflation subspace parameters are not set");
      bs=dfl.bs;
      nvec=VOLUME+FACE0*bs[0]+FACE1*bs[1]+FACE2*bs[2]+FACE3*bs[3];
      nvec/=(bs[0]*bs[1]*bs[2]*bs[3]);
      nvec*=dfl.Ns;

      if (type==4)
         return sizeof(complex)*(size_t)(nvec);
      else
         return sizeof(complex_dble)*(size_t)(nvec);
   }
}


static void alloc_wspace(int n,wspace_t *wsp,char *program)
{
   int k,iprms[1];
   int type,*nw;
   size_t fsize;
   float *f,*fm;
   double *fd,*fdm;
   void **w0;

   if (NPROC>1)
   {
      iprms[0]=n;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      error(iprms[0]!=n,1,program,"Parameter n is not global");
   }

   if (n==(*wsp).n)
      return;

   error((*wsp).nres!=0,1,program,"Fields are in use");

   if ((*wsp).n>0)
   {
      free((*wsp).nw);
      afree((*wsp).w0[0]);
      free((*wsp).w0);
   }

   (*wsp).n=n;
   (*wsp).nres=0;
   (*wsp).ires=0;
   (*wsp).nw=NULL;
   (*wsp).w0=NULL;
   (*wsp).w=NULL;

   if (n>0)
   {
      type=(*wsp).type;

      nw=malloc(n*sizeof(*nw));
      w0=malloc(2*n*sizeof(*w0));
      error((nw==NULL)||(w0==NULL),1,program,"Unable to allocate index arrays");

      fsize=fld_size(type);
      w0[0]=amalloc((size_t)(n)*fsize,ALIGN);
      error(w0[0]==NULL,1,program,"Unable to allocate field array");

      for (k=0;k<n;k++)
      {
         if (k>0)
            w0[k]=(void*)((char*)(w0[k-1])+fsize);

         nw[k]=0;
         w0[k+n]=NULL;
      }

      if ((type==2)||(type==4))
      {
         f=(float*)(w0[0]);
         fm=f+(fsize/sizeof(float))*(size_t)(n);

         for (;f<fm;f++)
            f[0]=0.0f;
      }
      else
      {
         fd=(double*)(w0[0]);
         fdm=fd+(fsize/sizeof(double))*(size_t)(n);

         for (;fd<fdm;fd++)
            fd[0]=0.0;
      }

      (*wsp).nw=nw;
      (*wsp).w0=w0;
      (*wsp).w=w0+n;
   }
}


static void **reserve_wsp(int n,wspace_t *wsp,char *program)
{
   int i,iprms[1];
   int nfld,nres,ires;
   void **w0,**w;

   if (NPROC>1)
   {
      iprms[0]=n;
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);
      error(iprms[0]!=n,1,program,"Parameter n is not global");
   }

   if (n>0)
   {
      nfld=(*wsp).n;
      nres=(*wsp).nres;
      ires=(*wsp).ires;
      w0=(*wsp).w0;
      w=(*wsp).w;

      error((nres+n)>nfld,1,program,
            "Requested too many fields (tot=%d,use=%d,req=%d)",nfld,nres,n);

      (*wsp).nres+=n;
      (*wsp).ires+=1;
      (*wsp).nw[ires]=n;

      for (i=nres;i<(nres+n);i++)
         w[i]=w0[i];

      return w+nres;
   }
   else
      return NULL;
}


static int release_wsp(wspace_t *wsp)
{
   int i,n,nres,ires,*nw;
   void **w;

   nres=(*wsp).nres;
   ires=(*wsp).ires;
   nw=(*wsp).nw;
   w=(*wsp).w;

   if (nres==0)
      return 0;
   else
   {
      ires-=1;
      n=nw[ires];
      nres-=n;

      (*wsp).nres=nres;
      (*wsp).ires=ires;
      nw[ires]=0;

      for (i=nres;i<(nres+n);i++)
         w[i]=NULL;

      return n;
   }
}


void alloc_wud(int n)
{
   alloc_wspace(n,&wud,"alloc_wud [wspace.c]");
}


su3_dble **reserve_wud(int n)
{
   return (su3_dble**)(reserve_wsp(n,&wud,"reserve_wud [wspace.c]"));
}


int release_wud(void)
{
   return release_wsp(&wud);
}


void alloc_wfd(int n)
{
   alloc_wspace(n,&wfd,"alloc_wfd [wspace.c]");
}


su3_alg_dble **reserve_wfd(int n)
{
   return (su3_alg_dble**)(reserve_wsp(n,&wfd,"reserve_wfd [wspace.c]"));
}


int release_wfd(void)
{
   return release_wsp(&wfd);
}


void alloc_ws(int n)
{
   alloc_wspace(n,&ws,"alloc_ws [wspace.c]");

   if (usews)
   {
      usews=0;
      wsd_uses_ws();
   }
}


spinor **reserve_ws(int n)
{
   return (spinor**)(reserve_wsp(n,&ws,"reserve_ws [wspace.c]"));
}


int release_ws(void)
{
   int n;

   n=release_wsp(&ws);
   error_loc((n>0)&&(usews)&&(iws[ws.ires]!=0),1,"release_ws [wspace.c]",
             "Incorrectly ordered ws and wsd release statements");

   return n;
}


void alloc_wsd(int n)
{
   if (usews==0)
      alloc_wspace(n,&wsd,"alloc_wsd [wspace.c]");
   else
      error(1,1,"alloc_wsd [wspace.c]",
            "wsd_uses_ws() has previously been called");
}


void wsd_uses_ws(void)
{
   int n,k,*nw;
   void **w;

   if (usews==0)
   {
      if (wsd.n==0)
      {
         if (iws!=NULL)
         {
            free(iws);
            free(wsd.nw);
            free(wsd.w);
         }

         iws=NULL;
         wsd.nres=0;
         wsd.ires=0;
         wsd.nw=NULL;
         wsd.w=NULL;
         n=ws.n/2;

         if (n>0)
         {
            iws=malloc(2*n*sizeof(*iws));
            nw=malloc(n*sizeof(*iws));
            w=malloc(n*sizeof(*w));
            error((iws==NULL)||(nw==NULL)||(w==NULL),1,"wsd_uses_ws [wspace.c]",
                  "Unable to allocate index arrays");

            for (k=0;k<(2*n);k++)
               iws[k]=0;

            for (k=0;k<n;k++)
            {
               nw[k]=0;
               w[k]=NULL;
            }

            wsd.nw=nw;
            wsd.w=w;
         }

         usews=1;
      }
      else
         error(1,1,"wsd_uses_ws [wspace.c]",
               "alloc_wsd() has previously been called");
   }
}


spinor_dble **reserve_wsd(int n)
{
   int i,nres,ires,*nw;
   void **w0,**w;

   if (n>0)
   {
      if (usews==0)
         return (spinor_dble**)(reserve_wsp(n,&wsd,"reserve_wsd [wspace.c]"));
      else
      {
         nres=wsd.nres;
         ires=wsd.ires;
         nw=wsd.nw;
         w=wsd.w;

         w0=reserve_wsp(2*n,&ws,"reserve_ws [wspace.c]");
         iws[ws.ires-1]=1;

         wsd.nres+=n;
         wsd.ires+=1;
         nw[ires]=n;
         w+=nres;

         for (i=0;i<n;i++)
            w[i]=w0[2*i];

         return (spinor_dble**)(w);
      }
   }
   else
      return NULL;
}


int release_wsd(void)
{
   int n;

   n=release_wsp(&wsd);

   if ((n>0)&&(usews))
   {
      iws[ws.ires-1]^=1;
      (void)(release_wsp(&ws));
   }

   return n;
}


void alloc_wv(int n)
{
   alloc_wspace(n,&wv,"alloc_wv [wspace.c]");
}


complex **reserve_wv(int n)
{
   return (complex**)(reserve_wsp(n,&wv,"reserve_wv [wspace.c]"));
}


int release_wv(void)
{
   return release_wsp(&wv);
}


void alloc_wvd(int n)
{
   alloc_wspace(n,&wvd,"alloc_wvd [wspace.c]");
}


complex_dble **reserve_wvd(int n)
{
   return (complex_dble**)(reserve_wsp(n,&wvd,"reserve_wvd [wspace.c]"));
}


int release_wvd(void)
{
   return release_wsp(&wvd);
}


size_t wsp_msize(void)
{
   size_t ntot;

   ntot=0;

   if (wud.n)
      ntot+=(size_t)(wud.n)*fld_size(0);
   if (wfd.n)
      ntot+=(size_t)(wfd.n)*fld_size(1);
   if (ws.n)
      ntot+=(size_t)(ws.n)*fld_size(2);
   if (wsd.n)
      ntot+=(size_t)(wsd.n)*fld_size(3);
   if (wv.n)
      ntot+=(size_t)(wv.n)*fld_size(4);
   if (wvd.n)
      ntot+=(size_t)(wvd.n)*fld_size(5);

   return ntot;
}


void print_wsp(void)
{
   int my_rank;
   double rt,r;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   rt=(double)(wsp_msize());

   if (rt!=0.0)
   {
      if (my_rank==0)
         printf("Contents and relative size of the workspaces:\n");

      if (wud.n)
      {
         r=100.0*(double)(wud.n)*(double)(fld_size(0))/rt;
         if (my_rank==0)
            printf("wud: nflds = %3d, size = %5.1f%%\n",wud.n,r);
      }

      if (wfd.n)
      {
         r=100.0*(double)(wfd.n)*(double)(fld_size(1))/rt;
         if (my_rank==0)
            printf("wfd: nflds = %3d, size = %5.1f%%\n",wfd.n,r);
      }

      if (ws.n)
      {
         r=100.0*(double)(ws.n)*(double)(fld_size(2))/rt;
         if (my_rank==0)
            printf("ws:  nflds = %3d, size = %5.1f%%\n",ws.n,r);
      }

      if (wsd.n)
      {
         r=100.0*(double)(wsd.n)*(double)(fld_size(3))/rt;
         if (my_rank==0)
            printf("wsd: nflds = %3d, size = %5.1f%%\n",wsd.n,r);
      }

      if (wv.n)
      {
         r=100.0*(double)(wv.n)*(double)(fld_size(4))/rt;
         if (my_rank==0)
            printf("wv:  nflds = %3d, size = %5.1f%%\n",wv.n,r);
      }

      if (wvd.n)
      {
         r=100.0*(double)(wvd.n)*(double)(fld_size(5))/rt;
         if (my_rank==0)
            printf("wvd: nflds = %3d, size = %5.1f%%\n",wvd.n,r);
      }

      if (my_rank==0)
         printf("\n");
   }
}
