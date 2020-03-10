
/*******************************************************************************
*
* File fldops.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Basic utility programs for observable fields.
*
*   void gather_fld(double *f,complex_dble *rf)
*     Assigns the array elements f[ipt[ix]], 0<=ix<VOLUME, to rf[ix].re
*     and sets rf[ix].im to zero.
*
*   void scatter_fld(complex_dble *rf,double *f)
*     Assigns the array elements rf[ix].re, 0<=ix<VOLUME, to f[ipt[ix]].
*
*   void apply_fft(int type,complex_dble *rf,complex_dble *rft)
*     Computes the Fourier (type=1) or the inverse Fourier (type=-1)
*     transform of the field rf and assigns the result to the field
*     rft. The field rf is unchanged on exit unless rft=rf (which is
*     permissible).
*
*   void convolute_flds(int *s,double *f,double *g,
*                       complex_dble *rf,complex_dble *rg,double *fg)
*     Computes the convolution fg=sum_y{f(x+s+y)*g(y)} of the fields f and
*     g translated by the integer four-vector s. No translation is applied
*     if s is set to NULL. The fields rf and rg are used as workspace. If
*     f=g it is permissible to set rf=rg, and setting fg to either f or g
*     is always permitted.
*
*   void shift_fld(int *s,double *f,complex_dble *rf,double *g)
*     Sets g(x)=f(x+s) at all points x. The field rf is used as workspace
*     and the translation is performed taking the periodicity of the lattice
*     into account. It is permissible to set g=f.
*
*   void copy_flds(double *f,double *g)
*     Assigns the field g to f.
*
*   void add_flds(double *f,double *g)
*     Assigns the pointwise sum f+g to f.
*
*   void mul_flds(double *f,double *g)
*     Assigns the pointwise product f*g to f.
*
*   void mulr_fld(double r,double *f)
*     Multiplies the field f by r.
*
* The programs in this module assume periodic boundary conditions in all
* directions. An error occurs if this is not the case. All programs act
* on global real or complex scalar fields.
*
* For the definition of the Fourier transform, see doc/dft.pdf and the
* the program files dft/fft.c and flags/dft4d_parms.c.
*
* The elements of the complex field arrays are assumed to be in lexicographic
* order, where rf[ix], for example, is the field value at the point with local
* coordinates (x0,x1,x2,x3) such that
*
*  ix=x3+x2*L3+x1*L2*L3+x0*L1*L2*L3.
*
* In the case of the real-valued fields, the standard ordering defined by
* the geometry routines is assumed (see main/README.global). The programs
* gather_fld() and scatter_fld() serve to move data from one type of field
* to the other.
*
* The programs in this module perform global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define FLDOPS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "dft.h"
#include "msfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int ids,init=0;
static int ls[4]={L0,L1,L2,L3};
static int ns[4]={N0,N1,N2,N3};
static complex_dble *sfct[4];


static void set_sfct(void)
{
   int mu,k;
   double pi,r,q;
   complex_dble *p;

   p=malloc((N0+N1+N2+N3)*sizeof(*p));
   error(p==NULL,1,"set_sfct [fldops.c]",
         "Unable to allocate auxiliary arrays");

   sfct[0]=p;
   p+=N0;
   sfct[1]=p;
   p+=N1;
   sfct[2]=p;
   p+=N2;
   sfct[3]=p;

   pi=4.0*atan(1.0);

   for (mu=0;mu<4;mu++)
   {
      r=(2.0*pi)/(double)(ns[mu]);

      for (k=0;k<ns[mu];k++)
      {
         q=(double)(k)*r;
         sfct[mu][k].re=cos(q);
         sfct[mu][k].im=sin(q);
      }
   }
}


static void mul_sfct(int *s,int *p,complex_dble *z)
{
   int mu,k;
   complex_dble w,*sf;

   for (mu=0;mu<4;mu++)
   {
      k=s[mu]*(cpr[mu]*ls[mu]+p[mu]);
      k=safe_mod(k,ns[mu]);

      sf=sfct[mu]+k;
      w.re=(*z).re*(*sf).re+(*z).im*(*sf).im;
      w.im=(*z).im*(*sf).re-(*z).re*(*sf).im;

      (*z).re=w.re;
      (*z).im=w.im;
   }
}


static void set_ids(void)
{
   int idp[4],nx[4];

   error_root(iup[0][0]==0,1,"set_ids [fldops.c]",
              "Geometry arrays are not set");
   error_root(bc_type()!=3,1,"set_ids [fldops.c]",
              "Improper boundary conditions (must be periodic)");

   idp[0]=set_dft_parms(EXP,N0,0,0);
   idp[1]=set_dft_parms(EXP,N1,0,0);
   idp[2]=set_dft_parms(EXP,N2,0,0);
   idp[3]=set_dft_parms(EXP,N3,0,0);

   nx[0]=L0;
   nx[1]=L1;
   nx[2]=L2;
   nx[3]=L3;

   ids=set_dft4d_parms(idp,nx,1);
   set_sfct();
   init=1;
}


void gather_fld(double *f,complex_dble *rf)
{
   int ix;

   if (init==0)
      set_ids();

   for (ix=0;ix<VOLUME;ix++)
   {
      rf[0].re=f[ipt[ix]];
      rf[0].im=0.0;
      rf+=1;
   }
}


void scatter_fld(complex_dble *rf,double *f)
{
   int ix;

   if (init==0)
      set_ids();

   for (ix=0;ix<VOLUME;ix++)
   {
      f[ipt[ix]]=rf[0].re;
      rf+=1;
   }
}


void apply_fft(int type,complex_dble *rf,complex_dble *rft)
{
   int iprms[1];

   if (init==0)
      set_ids();

   error_root((type!=1)&&(type!=-1),1,"apply_fft [fldops.c]",
              "Parameter type must be +1 or -1");

   if (NPROC>1)
   {
      iprms[0]=type;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=type,1,"apply_fft [fldops.c]",
            "Parameters are not global");
   }

   if (type==1)
      dft4d(ids,rf,rft);
   else
      inv_dft4d(ids,rf,rft);
}


void convolute_flds(int *s,double *f,double *g,
                    complex_dble *rf,complex_dble *rg,double *fg)
{
   int ip,iq,p[4];
   int iprms[4];
   complex_dble z;

   if (init==0)
      set_ids();

   if (NPROC>1)
   {
      if (s==NULL)
      {
         p[0]=0;
         p[1]=0;
         p[2]=0;
         p[3]=0;
      }
      else
      {
         p[0]=s[0];
         p[1]=s[1];
         p[2]=s[2];
         p[3]=s[3];
      }

      iprms[0]=p[0];
      iprms[1]=p[1];
      iprms[2]=p[2];
      iprms[3]=p[3];

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=p[0])||(iprms[1]!=p[1])||(iprms[2]!=p[2])||
            (iprms[3]!=p[3]),1,"convolute_flds [fldops.c]",
            "Parameters are not global");
   }

   gather_fld(f,rf);
   dft4d(ids,rf,rf);

   if (f!=g)
   {
      gather_fld(g,rg);
      dft4d(ids,rg,rg);
   }
   else
      rg=rf;

   if (s==NULL)
   {
      for (ip=0;ip<VOLUME;ip++)
      {
         z.re=rf[ip].re*rg[ip].re+rf[ip].im*rg[ip].im;
         z.im=rf[ip].im*rg[ip].re-rf[ip].re*rg[ip].im;

         rf[ip].re=z.re;
         rf[ip].im=z.im;
      }
   }
   else
   {
      for (ip=0;ip<VOLUME;ip++)
      {
         iq=ip;
         p[3]=iq%L3;
         iq/=L3;
         p[2]=iq%L2;
         iq/=L2;
         p[1]=iq%L1;
         iq/=L1;
         p[0]=iq;

         z.re=rf[ip].re*rg[ip].re+rf[ip].im*rg[ip].im;
         z.im=rf[ip].im*rg[ip].re-rf[ip].re*rg[ip].im;

         mul_sfct(s,p,&z);

         rf[ip].re=z.re;
         rf[ip].im=z.im;
      }
   }

   inv_dft4d(ids,rf,rf);
   scatter_fld(rf,fg);
}


void shift_fld(int *s,double *f,complex_dble *rf,double *g)
{
   int ip,iq,p[4];
   int iprms[4];

   if (init==0)
      set_ids();

   if (NPROC>1)
   {
      iprms[0]=s[0];
      iprms[1]=s[1];
      iprms[2]=s[2];
      iprms[3]=s[3];

      MPI_Bcast(iprms,4,MPI_INT,0,MPI_COMM_WORLD);

      error((iprms[0]!=s[0])||(iprms[1]!=s[1])||(iprms[2]!=s[2])||
            (iprms[3]!=s[3]),1,"shift_fld [fldops.c]",
            "Parameters are not global");
   }

   gather_fld(f,rf);
   dft4d(ids,rf,rf);

   for (ip=0;ip<VOLUME;ip++)
   {
      iq=ip;
      p[3]=iq%L3;
      iq/=L3;
      p[2]=iq%L2;
      iq/=L2;
      p[1]=iq%L1;
      iq/=L1;
      p[0]=iq;

      mul_sfct(s,p,rf+ip);
   }

   inv_dft4d(ids,rf,rf);
   scatter_fld(rf,g);
}


void copy_flds(double *f,double *g)
{
   int ix;

   for (ix=0;ix<VOLUME;ix++)
      f[ix]=g[ix];
}


void add_flds(double *f,double *g)
{
   int ix;

   for (ix=0;ix<VOLUME;ix++)
      f[ix]+=g[ix];
}


void mul_flds(double *f,double *g)
{
   int ix;

   for (ix=0;ix<VOLUME;ix++)
      f[ix]*=g[ix];
}


void mulr_fld(double r,double *f)
{
   int ix;

   for (ix=0;ix<VOLUME;ix++)
      f[ix]*=r;
}
