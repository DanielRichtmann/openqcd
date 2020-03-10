
/*******************************************************************************
*
* File small_dft.c
*
* Copyright (C) 2015, 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Small discrete Fourier transforms.
*
*   void small_dft(int s,int n,int nfc,complex_dble *w,complex_dble **f,
*                  complex_dble **ft)
*     Computes the Fourier transforms ft of the functions w*f (see the notes).
*
* The discrete Fourier transform of n values f[0],..,f[n-1] is defined by
*
*  ft[k]=sum_{x=0}^{n-1} exp(s*i*2*pi*k*x/n)*w[x]*f[x]
*
* where s=+1 or s=-1 and k=0,1,..,n-1.
*
* The arguments of the program small_dft() are
*
*  s          Exponent sign.
*
*  n          Number of function values. Currently n must be 1,2,3,4 or 5.
*
*  nfc        Number of functions to be transformed.
*
*  w[x]       Weight factor (x=0,1,..,n-1).
*
*  f[x][i]    Function values of the i'th function (x=0,1,..,n-1,
*             i=0,1,..,nfc-1).
*
*  ft[k][i]   Calculated Fourier transform of the i'th function
*             (k=0,1,2,..,n-1). It is permissible to set ft=f.
*
* See appendix A of the notes
*
*  M. Luescher: "Discrete Fourier transform", January 2015,
*
* for further information.
*
* This program does not perform any communications and can be locally called.
*
*******************************************************************************/

#define SMALL_DFT_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "dft.h"

static int na=0,ns=0,init=0;
static double u1,v1,u2,v2,u3,v3,pi;
static complex_dble *fs,*ekx;


static void alloc_fs(int n)
{
   if (na>0)
      free(fs);

   fs=malloc(2*n*sizeof(*fs));
   error_loc(fs==NULL,1,"alloc_fs [small_dft.c]",
             "Unable to allocate auxiliary arrays");

   ekx=fs+n;
   na=n;
}


static void set_ekx(int n)
{
   int k;
   double r;

   if (ns==0)
      pi=4.0*atan(1.0);

   r=2.0*pi/(double)(n);

   for (k=0;k<n;k++)
   {
      ekx[k].re=cos((double)(k)*r);
      ekx[k].im=sin((double)(k)*r);
   }

   ns=n;
}


static void set_fs(int n,int i,complex_dble **f,complex_dble *w)
{
   int x;

   for (x=0;x<n;x++)
   {
      fs[x].re=w[x].re*f[x][i].re-w[x].im*f[x][i].im;
      fs[x].im=w[x].re*f[x][i].im+w[x].im*f[x][i].re;
   }
}


static void set_ft(int s,int n,int i,complex_dble **ft)
{
   int k,x,p;
   complex_dble sm;

   for (k=0;k<n;k++)
   {
      sm.re=0.0;
      sm.im=0.0;

      for (x=0;x<n;x++)
      {
         p=(k*x)%n;

         if (s==1)
         {
            sm.re+=ekx[p].re*fs[x].re-ekx[p].im*fs[x].im;
            sm.im+=ekx[p].re*fs[x].im+ekx[p].im*fs[x].re;
         }
         else
         {
            sm.re+=ekx[p].re*fs[x].re+ekx[p].im*fs[x].im;
            sm.im+=ekx[p].re*fs[x].im-ekx[p].im*fs[x].re;
         }
      }

      ft[k][i].re=sm.re;
      ft[k][i].im=sm.im;
   }
}


static void set_wfact(void)
{
   u1=-0.5;
   v1=0.5*sqrt(3.0);

   u2=1.0/(1.0+sqrt(5.0));
   v2=sqrt((5.0+sqrt(5.0))/8.0);

   u3=-u2-0.5;
   v3=2.0*u2*v2;

   init=1;
}


static void small_dft1(int s,int nfc,complex_dble *w,complex_dble **f,
                       complex_dble **ft)
{
   int i;
   complex_dble *f0,*ft0;
   complex_dble wf0;

   f0=f[0];
   ft0=ft[0];

   for (i=0;i<nfc;i++)
   {
      wf0.re=w[0].re*f0[0].re-w[0].im*f0[0].im;
      wf0.im=w[0].re*f0[0].im+w[0].im*f0[0].re;

      ft0[0].re=wf0.re;
      ft0[0].im=wf0.im;

      f0+=1;
      ft0+=1;
   }
}


static void small_dft2(int s,int nfc,complex_dble *w,complex_dble **f,
                       complex_dble **ft)
{
   int i;
   complex_dble *f0,*f1,*ft0,*ft1;
   complex_dble wf0,wf1;

   f0=f[0];
   f1=f[1];
   ft0=ft[0];
   ft1=ft[1];

   for (i=0;i<nfc;i++)
   {
      wf0.re=w[0].re*f0[0].re-w[0].im*f0[0].im;
      wf0.im=w[0].re*f0[0].im+w[0].im*f0[0].re;

      wf1.re=w[1].re*f1[0].re-w[1].im*f1[0].im;
      wf1.im=w[1].re*f1[0].im+w[1].im*f1[0].re;

      ft0[0].re=wf0.re+wf1.re;
      ft0[0].im=wf0.im+wf1.im;

      ft1[0].re=wf0.re-wf1.re;
      ft1[0].im=wf0.im-wf1.im;

      f0+=1;
      f1+=1;
      ft0+=1;
      ft1+=1;
   }
}


static void small_dft3(int s,int nfc,complex_dble *w,complex_dble **f,
                       complex_dble **ft)
{
   int i;
   double sv1;
   complex_dble *f0,*f1,*f2,*ft0,*ft1,*ft2;
   complex_dble wf0,wf1,wf2;
   complex_dble z1p2,z1m2;

   f0=f[0];
   f1=f[1];
   f2=f[2];

   ft0=ft[0];
   ft1=ft[1];
   ft2=ft[2];

   sv1=(double)(s)*v1;

   for (i=0;i<nfc;i++)
   {
      wf0.re=w[0].re*f0[0].re-w[0].im*f0[0].im;
      wf0.im=w[0].re*f0[0].im+w[0].im*f0[0].re;

      wf1.re=w[1].re*f1[0].re-w[1].im*f1[0].im;
      wf1.im=w[1].re*f1[0].im+w[1].im*f1[0].re;

      wf2.re=w[2].re*f2[0].re-w[2].im*f2[0].im;
      wf2.im=w[2].re*f2[0].im+w[2].im*f2[0].re;

      z1p2.re=wf1.re+wf2.re;
      z1p2.im=wf1.im+wf2.im;

      z1m2.re=wf1.re-wf2.re;
      z1m2.im=wf1.im-wf2.im;

      ft0[0].re=wf0.re+z1p2.re;
      ft0[0].im=wf0.im+z1p2.im;

      z1p2.re=wf0.re+u1*z1p2.re;
      z1p2.im=wf0.im+u1*z1p2.im;

      z1m2.re*=sv1;
      z1m2.im*=sv1;

      ft1[0].re=z1p2.re-z1m2.im;
      ft1[0].im=z1p2.im+z1m2.re;

      ft2[0].re=z1p2.re+z1m2.im;
      ft2[0].im=z1p2.im-z1m2.re;

      f0+=1;
      f1+=1;
      f2+=1;

      ft0+=1;
      ft1+=1;
      ft2+=1;
   }
}


static void small_dft4(int s,int nfc,complex_dble *w,complex_dble **f,
                       complex_dble **ft)
{
   int i;
   complex_dble *f0,*f1,*f2,*f3,*ft0,*ft1,*ft2,*ft3;
   complex_dble wf0,wf1,wf2,wf3;
   complex_dble z1,z2,z3,z4;

   f0=f[0];
   f1=f[1];
   f2=f[2];
   f3=f[3];

   ft0=ft[0];
   ft2=ft[2];

   if (s==1)
   {
      ft1=ft[1];
      ft3=ft[3];
   }
   else
   {
      ft1=ft[3];
      ft3=ft[1];
   }

   for (i=0;i<nfc;i++)
   {
      wf0.re=w[0].re*f0[0].re-w[0].im*f0[0].im;
      wf0.im=w[0].re*f0[0].im+w[0].im*f0[0].re;

      wf1.re=w[1].re*f1[0].re-w[1].im*f1[0].im;
      wf1.im=w[1].re*f1[0].im+w[1].im*f1[0].re;

      wf2.re=w[2].re*f2[0].re-w[2].im*f2[0].im;
      wf2.im=w[2].re*f2[0].im+w[2].im*f2[0].re;

      wf3.re=w[3].re*f3[0].re-w[3].im*f3[0].im;
      wf3.im=w[3].re*f3[0].im+w[3].im*f3[0].re;

      z1.re=wf0.re+wf2.re;
      z1.im=wf0.im+wf2.im;

      z2.re=wf0.re-wf2.re;
      z2.im=wf0.im-wf2.im;

      z3.re=wf1.re+wf3.re;
      z3.im=wf1.im+wf3.im;

      z4.re=wf1.re-wf3.re;
      z4.im=wf1.im-wf3.im;

      ft0[0].re=z1.re+z3.re;
      ft0[0].im=z1.im+z3.im;

      ft2[0].re=z1.re-z3.re;
      ft2[0].im=z1.im-z3.im;

      ft1[0].re=z2.re-z4.im;
      ft1[0].im=z2.im+z4.re;

      ft3[0].re=z2.re+z4.im;
      ft3[0].im=z2.im-z4.re;

      f0+=1;
      f1+=1;
      f2+=1;
      f3+=1;

      ft0+=1;
      ft1+=1;
      ft2+=1;
      ft3+=1;
   }
}


static void small_dft5(int s,int nfc,complex_dble *w,complex_dble **f,
                       complex_dble **ft)
{
   int i;
   double sv2,sv3;
   complex_dble *f0,*f1,*f2,*f3,*f4;
   complex_dble *ft0,*ft1,*ft2,*ft3,*ft4;
   complex_dble wf0,wf1,wf2,wf3,wf4;
   complex_dble z1p4,z1m4,z2p3,z2m3;

   f0=f[0];
   f1=f[1];
   f2=f[2];
   f3=f[3];
   f4=f[4];

   ft0=ft[0];
   ft1=ft[1];
   ft2=ft[2];
   ft3=ft[3];
   ft4=ft[4];

   sv2=(double)(s)*v2;
   sv3=(double)(s)*v3;

   for (i=0;i<nfc;i++)
   {
      wf0.re=w[0].re*f0[0].re-w[0].im*f0[0].im;
      wf0.im=w[0].re*f0[0].im+w[0].im*f0[0].re;

      wf1.re=w[1].re*f1[0].re-w[1].im*f1[0].im;
      wf1.im=w[1].re*f1[0].im+w[1].im*f1[0].re;

      wf2.re=w[2].re*f2[0].re-w[2].im*f2[0].im;
      wf2.im=w[2].re*f2[0].im+w[2].im*f2[0].re;

      wf3.re=w[3].re*f3[0].re-w[3].im*f3[0].im;
      wf3.im=w[3].re*f3[0].im+w[3].im*f3[0].re;

      wf4.re=w[4].re*f4[0].re-w[4].im*f4[0].im;
      wf4.im=w[4].re*f4[0].im+w[4].im*f4[0].re;

      z1p4.re=wf1.re+wf4.re;
      z1p4.im=wf1.im+wf4.im;

      z1m4.re=wf1.re-wf4.re;
      z1m4.im=wf1.im-wf4.im;

      z2p3.re=wf2.re+wf3.re;
      z2p3.im=wf2.im+wf3.im;

      z2m3.re=wf2.re-wf3.re;
      z2m3.im=wf2.im-wf3.im;

      ft0[0].re=wf0.re+z1p4.re+z2p3.re;
      ft0[0].im=wf0.im+z1p4.im+z2p3.im;

      ft1[0].re=wf0.re+u2*z1p4.re+u3*z2p3.re;
      ft1[0].im=wf0.im+u2*z1p4.im+u3*z2p3.im;

      ft2[0].re=wf0.re+u3*z1p4.re+u2*z2p3.re;
      ft2[0].im=wf0.im+u3*z1p4.im+u2*z2p3.im;

      z2p3.re=sv3*z2m3.re+sv2*z1m4.re;
      z2p3.im=sv3*z2m3.im+sv2*z1m4.im;

      z1p4.re=sv3*z1m4.re-sv2*z2m3.re;
      z1p4.im=sv3*z1m4.im-sv2*z2m3.im;

      ft4[0].re=ft1[0].re+z2p3.im;
      ft4[0].im=ft1[0].im-z2p3.re;

      ft3[0].re=ft2[0].re+z1p4.im;
      ft3[0].im=ft2[0].im-z1p4.re;

      ft1[0].re-=z2p3.im;
      ft1[0].im+=z2p3.re;

      ft2[0].re-=z1p4.im;
      ft2[0].im+=z1p4.re;

      f0+=1;
      f1+=1;
      f2+=1;
      f3+=1;
      f4+=1;

      ft0+=1;
      ft1+=1;
      ft2+=1;
      ft3+=1;
      ft4+=1;
   }
}


static void (*sdft[6])(int s,int nfc,complex_dble *w,complex_dble **f,
                       complex_dble **ft)={NULL,small_dft1,small_dft2,
                                           small_dft3,small_dft4,small_dft5};


void small_dft(int s,int n,int nfc,complex_dble *w,complex_dble **f,
               complex_dble **ft)
{
   int i;

   error_loc(((s!=1)&&(s!=-1))||(n<1)||(nfc<1),1,"small_dft [small_dft.c]",
             "Arguments are out of range");

   if (n<=5)
   {
      if (init==0)
         set_wfact();

      sdft[n](s,nfc,w,f,ft);
   }
   else
   {
      if (n>na)
         alloc_fs(n);
      if (n!=ns)
         set_ekx(n);

      for (i=0;i<nfc;i++)
      {
         set_fs(n,i,f,w);
         set_ft(s,n,i,ft);
      }
   }
}
