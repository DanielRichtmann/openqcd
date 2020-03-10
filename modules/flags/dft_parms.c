
/*******************************************************************************
*
* File dft_parms.c
*
* Copyright (C) 2015, 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* DFT parameter data base.
*
*   int set_dft_parms(dft_type_t type,int n,int b,int c)
*     Initializes a new instance of a DFT parameter set for the Fourier
*     transformations of a specific type (see the notes). The program
*     returns the id of the parameter set in the data base.
*
*   dft_parms_t *dft_parms(int id)
*     Returns the DFT parameter set with the given id. If there is no
*     such parameter set, the program returns NULL.
*
* The discrete Fourier transform operates on arrays of double-precision
* complex numbers. Different types of Fourier transforms are implemented
* as described in the notes
*
*  M. Luescher: "Discrete Fourier transform", January 2015, doc/dft.pdf.
*
* The first element of a structure of type dft_parms_t is
*
*  type       Transformation type = EXP, SIN or COS,
*
* where EXP stands for the normal Fourier transform [eq.(2.4) in the notes]
* and SIN and COS for the sine and cosine transformations [eqs.(3.5),(3.6)].
*
* The interpretation of the other parameters in the structure depends on the
* transformation type.
*
* If type=EXP, the parameters are:
*
*  n          Number of function values.
*
*  b,c        One-bit integers specifying the momentum and coordinate shifts
*             in the Fourier transform.
*
*  r[x]       Reordering array (0<=x<n; see section 4.2 of the notes).
*
*  w[x]       Set to exp{i*2*pi*x/n}, 0<=x<=n.
*
*  wb[x]      Set to exp{i*pi*b*r[x]/n}, 0<=x<n.
*
*  wc[k]      Set to exp{i*pi*(k+b/2)*c/n}, 0<=k<n.
*
*  iwb[x]     Set to exp{-i*pi*b*(x+c/2)/n}, 0<=x<n.
*
*  iwc[k]     Set to exp{-i*pi*c*r[k]/n}/n, 0<=k<n.
*
* If type=SIN or type=COS, the parameters are:
*
*  n,b,c      Parameters of the space K^{bcd}_n of functions to which
*             the transformation is going to be applied, where d=1 if
*             type=SIN and d=0 if type=COS.
*
*  r[x]       Reordering array for the associated EXP Fourier transform
*             (0<=x<2*n; see section 4.2 of the notes).
*
*  w[x]       Set to exp{i*2*pi*x/(2*n)}, 0<=x<=2*n.
*
*  wb[x]      Set to exp{i*pi*b*r[x]/(2*n)}*chi(r[x],n,b,c,d), 0<=x<2*n.
*
*  wc[k]      Set to exp{i*pi*(k+b/2)*c/(2*n)}, 0<=k<2*n.
*
*  iwb[x]     Set to exp{-i*pi*b*(x+c/2)/(2*n)}, 0<=k<2*n.
*
*  iwc[k]     Set to exp{-i*pi*c*r[k]/(2*n)}*chi(r[k],n,c,b,d)/(2*n), 0<=k<2*n.
*
* Here
*
*  chi(x,n,b,c,d) =  0 if (x=0)&(c=0)&(d=1) or (x=n)&(c=0)&((b+d)=1),
*
*                 = -1 if (x>=n)&((b+d)=1),
*
*                 =  1 otherwise.
*
* The function spaces K^{bcd}_n are defined in the notes.
*
* The program set_dft_parms() performs global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define DFT_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

static int mxs=0,ids=0;
static int ns=0,*rs;
static dft_parms_t **dpa;


static void alloc_dpa(void)
{
   int i;
   dft_parms_t **dd,*d;

   error(mxs>(INT_MAX-16),1,"alloc_dpa [dft_parms.c]",
         "Unable to allocate dft_parms_t structures");

   dd=malloc((mxs+16)*sizeof(*dd));
   d=malloc(16*sizeof(*d));
   error((dd==NULL)||(d==NULL),2,"alloc_dpa [dft_parms.c]",
         "Unable to allocate dft_parms_t structures");

   for (i=0;i<mxs;i++)
      dd[i]=dpa[i];

   for (i=0;i<16;i++)
   {
      d[i].n=0;
      dd[mxs+i]=d+i;
   }

   if (mxs>0)
      free(dpa);

   dpa=dd;
   mxs+=16;
}


static void step_r(int n,int *r)
{
   int i,nh;

   if ((n>4)&&((n%2)==0))
   {
      nh=n/2;

      for (i=0;i<nh;i++)
      {
         rs[i]=r[2*i];
         rs[nh+i]=r[2*i+1];
      }

      for (i=0;i<n;i++)
         r[i]=rs[i];

      step_r(nh,r);
      step_r(nh,r+nh);
   }
}


static void set_r(dft_parms_t *dp)
{
   int i,n,*r;

   n=(*dp).n;
   r=(*dp).r;

   if ((*dp).type!=EXP)
      n*=2;

   for (i=0;i<n;i++)
      r[i]=i;

   step_r(n,r);
}


static void set_w(dft_parms_t *dp)
{
   int n,nh,k;
   double r0;
   complex_dble *w;

   n=(*dp).n;
   w=(*dp).w;

   if ((*dp).type!=EXP)
      n*=2;
   nh=n/2;
   r0=8.0*atan(1.0)/(double)(n);

   for (k=0;k<=n;k++)
   {
      if (k<=nh)
      {
         w[k].re=cos((double)(k)*r0);
         w[k].im=sin((double)(k)*r0);
      }
      else
      {
         w[k].re=cos((double)(k-n)*r0);
         w[k].im=sin((double)(k-n)*r0);
      }
   }
}


static double chi(int x,int n,int b,int c,int d)
{
   if (((x==0)&&(c==0)&&(d==1))||
       ((x==n)&&(c==0)&&((b+d)==1)))
      return 0.0;
   else if ((x>=n)&&((b+d)==1))
      return -1.0;
   else
      return 1.0;
}


static void set_wb(dft_parms_t *dp)
{
   int n,b,c,d,*r;
   int k,j;
   double r0,r1;
   complex_dble *wb,*iwb;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;
   d=((*dp).type==SIN);
   r=(*dp).r;
   wb=(*dp).wb;
   iwb=(*dp).iwb;

   if ((*dp).type!=EXP)
      n*=2;

   r0=4.0*atan(1.0)/(double)(n);

   for (k=0;k<n;k++)
   {
      if (b==1)
      {
         iwb[k].re=cos(r0*(double)(k));
         iwb[k].im=sin(r0*(double)(k));
      }
      else
      {
         iwb[k].re=1.0;
         iwb[k].im=0.0;
      }
   }

   for (k=0;k<n;k++)
   {
      j=r[k];
      if ((*dp).type!=EXP)
         r1=chi(j,n/2,b,c,d);
      else
         r1=1.0;
      wb[k].re=r1*iwb[j].re;
      wb[k].im=r1*iwb[j].im;
   }

   r0=2.0*atan(1.0)/(double)(n);

   for (k=0;k<n;k++)
   {
      if (b==1)
      {
         iwb[k].re=cos(r0*(double)(2*k+c));
         iwb[k].im=-sin(r0*(double)(2*k+c));
      }
      else
      {
         iwb[k].re=1.0;
         iwb[k].im=0.0;
      }
   }
}


static void set_wc(dft_parms_t *dp)
{
   int n,b,c,d,*r;
   int k,j;
   double r0,r1;
   complex_dble *wc,*iwc;

   n=(*dp).n;
   b=(*dp).b;
   c=(*dp).c;
   d=((*dp).type==SIN);
   r=(*dp).r;
   wc=(*dp).wc;
   iwc=(*dp).iwc;

   if ((*dp).type!=EXP)
      n*=2;

   r0=4.0*atan(1.0)/(double)(n);

   for (k=0;k<n;k++)
   {
      if (c==1)
      {
         wc[k].re=cos(r0*(double)(k));
         wc[k].im=-sin(r0*(double)(k));
      }
      else
      {
         wc[k].re=1.0;
         wc[k].im=0.0;
      }
   }

   r0=1.0/(double)(n);

   for (k=0;k<n;k++)
   {
      j=r[k];
      if ((*dp).type!=EXP)
         r1=chi(j,n/2,c,b,d);
      else
         r1=1.0;
      iwc[k].re=r0*r1*wc[j].re;
      iwc[k].im=r0*r1*wc[j].im;
   }

   r0=2.0*atan(1.0)/(double)(n);

   for (k=0;k<n;k++)
   {
      if (c==1)
      {
         wc[k].re=cos(r0*(double)(2*k+b));
         wc[k].im=sin(r0*(double)(2*k+b));
      }
      else
      {
         wc[k].re=1.0;
         wc[k].im=0.0;
      }
   }
}


static void set_rw(dft_parms_t *dp)
{
   int n,*r;
   complex_dble *w;

   n=(*dp).n;

   if ((*dp).type!=EXP)
      n*=2;

   r=malloc(n*sizeof(*r));
   w=amalloc((5*n+1)*sizeof(*w),4);

   if (n>ns)
   {
      if (ns>0)
         free(rs);

      rs=malloc(n*sizeof(*rs));
      ns=n;
   }

   error((r==NULL)||(w==NULL)||(rs==NULL),1,"set_rw [dft_parms.c]",
         "Unable to allocate arrays in new dft parameter structure");

   (*dp).r=r;
   (*dp).w=w;
   w+=(n+1);
   (*dp).wb=w;
   w+=n;
   (*dp).wc=w;
   w+=n;
   (*dp).iwb=w;
   w+=n;
   (*dp).iwc=w;

   set_r(dp);
   set_w(dp);
   set_wb(dp);
   set_wc(dp);
}


int set_dft_parms(dft_type_t type,int n,int b,int c)
{
   int id,iprms[4];
   dft_parms_t *dp;

   error_root((type==DFT_TYPES)||(n<1)||(b<0)||(b>1)||(c<0)||(c>1),1,
              "set_dft_parms [dft_parms.c]","Parameters are out of range");

   error_root((type==SIN)&&(n==1)&&(b==0)&&(c==0),1,
              "set_dft_parms [dft_parms.c]","Chosen function space is empty");

   if (NPROC>1)
   {
      iprms[0]=(int)(type);
      iprms[1]=n;
      iprms[2]=b;
      iprms[3]=c;

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=(int)(type))||(iprms[1]!=n)||(iprms[2]!=b)||
            (iprms[3]!=c),1,"set_dft_parms [dft_parms.c]",
            "Parameters are not global");
   }

   if (ids==mxs)
      alloc_dpa();

   id=ids;
   ids+=1;
   dp=dpa[id];

   (*dp).type=type;
   (*dp).n=n;
   (*dp).b=b;
   (*dp).c=c;

   set_rw(dp);

   return id;
}


dft_parms_t *dft_parms(int id)
{
   if ((id>=0)&&(id<mxs)&&(dpa[id][0].n!=0))
      return dpa[id];
   else
      return NULL;
}
