
/*******************************************************************************
*
* File unorm.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to the uniform norm of spinor fields.
*
*   float unorm(int vol,int icom,spinor *s)
*     Returns the uniform norm of the single-precision spinor field s. The
*     norm is maximized over all MPI processes if (and only if) icom=1.
*
*   double unorm_dble(int vol,int icom,spinor_dble *sd)
*     Returns the uniform norm of the double-precision spinor field sd. The
*     norm is maximized over all MPI processes if (and only if) icom=1.
*
* The uniform norm of a spinor field psi(x) is defined by
*
*   ||psi||_oo = max_x{||psi(x)||},
*
* where ||s|| denotes the 2-norm of the spinor s.
*
* All programs in this module operate on spinor fields, whose base address
* is passed through the arguments. The length of the arrays is specified by
* the parameter vol.
*
* If SSE instructions are used, the fields must be aligned to a 16 byte
* boundary.
*
*******************************************************************************/

#define UNORM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "sflds.h"
#include "global.h"

#if (defined x64)
#include "sse2.h"

static float loc_unorm(int vol,spinor *s)
{
   float nrm;
   spinor *sm;

   __asm__ __volatile__ ("xorps %%xmm7, %%xmm7"
                         :
                         :
                         :
                         "xmm7");

   sm=s+vol;

   for (;s<sm;s++)
   {
      _sse_spinor_load(*s);

      __asm__ __volatile__ ("mulps %%xmm0, %%xmm0 \n\t"
                            "mulps %%xmm1, %%xmm1 \n\t"
                            "mulps %%xmm2, %%xmm2 \n\t"
                            "mulps %%xmm3, %%xmm3 \n\t"
                            "mulps %%xmm4, %%xmm4 \n\t"
                            "mulps %%xmm5, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      __asm__ __volatile__ ("addps %%xmm0, %%xmm1 \n\t"
                            "addps %%xmm2, %%xmm3 \n\t"
                            "addps %%xmm4, %%xmm5 \n\t"
                            "addps %%xmm1, %%xmm5 \n\t"
                            "addps %%xmm3, %%xmm5 \n\t"
                            "movshdup %%xmm5, %%xmm6 \n\t"
                            "addps %%xmm5, %%xmm6 \n\t"
                            "movhlps %%xmm6, %%xmm4 \n\t"
                            "addss %%xmm6, %%xmm4 \n\t"
                            "maxss %%xmm4, %%xmm7"
                            :
                            :
                            :
                            "xmm1", "xmm3", "xmm4",
                            "xmm5", "xmm6", "xmm7");
   }

   __asm__ __volatile__ ("movss %%xmm7, %0"
                         :
                         "=m" (nrm));

   return nrm;
}


static double loc_unorm_dble(int vol,spinor_dble *sd)
{
   double nrm;
   spinor_dble *sm;

   __asm__ __volatile__ ("xorpd %%xmm7, %%xmm7"
                         :
                         :
                         :
                         "xmm7");

   sm=sd+vol;

   for (;sd<sm;sd++)
   {
      _sse_load_dble((*sd).c1);
      _sse_load_up_dble((*sd).c2);

      __asm__ __volatile__ ("mulpd %%xmm0, %%xmm0 \n\t"
                            "mulpd %%xmm1, %%xmm1 \n\t"
                            "mulpd %%xmm2, %%xmm2 \n\t"
                            "mulpd %%xmm3, %%xmm3 \n\t"
                            "mulpd %%xmm4, %%xmm4 \n\t"
                            "mulpd %%xmm5, %%xmm5 \n\t"
                            "addpd %%xmm0, %%xmm3 \n\t"
                            "addpd %%xmm1, %%xmm4 \n\t"
                            "addpd %%xmm2, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      _sse_load_dble((*sd).c3);

      __asm__ __volatile__ ("mulpd %%xmm0, %%xmm0 \n\t"
                            "mulpd %%xmm1, %%xmm1 \n\t"
                            "mulpd %%xmm2, %%xmm2 \n\t"
                            "addpd %%xmm0, %%xmm3 \n\t"
                            "addpd %%xmm1, %%xmm4 \n\t"
                            "addpd %%xmm2, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      _sse_load_dble((*sd).c4);

      __asm__ __volatile__ ("mulpd %%xmm0, %%xmm0 \n\t"
                            "mulpd %%xmm1, %%xmm1 \n\t"
                            "mulpd %%xmm2, %%xmm2 \n\t"
                            "addpd %%xmm0, %%xmm3 \n\t"
                            "addpd %%xmm1, %%xmm4 \n\t"
                            "addpd %%xmm2, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2",
                            "xmm3", "xmm4", "xmm5");

      __asm__ __volatile__ ("addpd %%xmm3, %%xmm5 \n\t"
                            "addpd %%xmm4, %%xmm5 \n\t"
                            "movhlps %%xmm5, %%xmm6 \n\t"
                            "addsd %%xmm5, %%xmm6 \n\t"
                            "maxsd %%xmm6, %%xmm7"
                            :
                            :
                            :
                            "xmm5", "xmm6", "xmm7");
   }

   __asm__ __volatile__ ("movsd %%xmm7, %0"
                         :
                         "=m" (nrm));

   return nrm;
}

#else

static float loc_unorm(int vol,spinor *s)
{
   float nrm,ns;
   spinor *sm;

   nrm=0.0f;
   sm=s+vol;

   for (;s<sm;s++)
   {
      ns=
         (*s).c1.c1.re*(*s).c1.c1.re+(*s).c1.c1.im*(*s).c1.c1.im+
         (*s).c1.c2.re*(*s).c1.c2.re+(*s).c1.c2.im*(*s).c1.c2.im+
         (*s).c1.c3.re*(*s).c1.c3.re+(*s).c1.c3.im*(*s).c1.c3.im;

      ns+=
         (*s).c2.c1.re*(*s).c2.c1.re+(*s).c2.c1.im*(*s).c2.c1.im+
         (*s).c2.c2.re*(*s).c2.c2.re+(*s).c2.c2.im*(*s).c2.c2.im+
         (*s).c2.c3.re*(*s).c2.c3.re+(*s).c2.c3.im*(*s).c2.c3.im;

      ns+=
         (*s).c3.c1.re*(*s).c3.c1.re+(*s).c3.c1.im*(*s).c3.c1.im+
         (*s).c3.c2.re*(*s).c3.c2.re+(*s).c3.c2.im*(*s).c3.c2.im+
         (*s).c3.c3.re*(*s).c3.c3.re+(*s).c3.c3.im*(*s).c3.c3.im;

      ns+=
         (*s).c4.c1.re*(*s).c4.c1.re+(*s).c4.c1.im*(*s).c4.c1.im+
         (*s).c4.c2.re*(*s).c4.c2.re+(*s).c4.c2.im*(*s).c4.c2.im+
         (*s).c4.c3.re*(*s).c4.c3.re+(*s).c4.c3.im*(*s).c4.c3.im;

      if (ns>nrm)
         nrm=ns;
   }

   return nrm;
}


static double loc_unorm_dble(int vol,spinor_dble *sd)
{
   double nrm,ns;
   spinor_dble *sm;

   nrm=0.0;
   sm=sd+vol;

   for (;sd<sm;sd++)
   {
      ns=
         (*sd).c1.c1.re*(*sd).c1.c1.re+(*sd).c1.c1.im*(*sd).c1.c1.im+
         (*sd).c1.c2.re*(*sd).c1.c2.re+(*sd).c1.c2.im*(*sd).c1.c2.im+
         (*sd).c1.c3.re*(*sd).c1.c3.re+(*sd).c1.c3.im*(*sd).c1.c3.im;

      ns+=
         (*sd).c2.c1.re*(*sd).c2.c1.re+(*sd).c2.c1.im*(*sd).c2.c1.im+
         (*sd).c2.c2.re*(*sd).c2.c2.re+(*sd).c2.c2.im*(*sd).c2.c2.im+
         (*sd).c2.c3.re*(*sd).c2.c3.re+(*sd).c2.c3.im*(*sd).c2.c3.im;

      ns+=
         (*sd).c3.c1.re*(*sd).c3.c1.re+(*sd).c3.c1.im*(*sd).c3.c1.im+
         (*sd).c3.c2.re*(*sd).c3.c2.re+(*sd).c3.c2.im*(*sd).c3.c2.im+
         (*sd).c3.c3.re*(*sd).c3.c3.re+(*sd).c3.c3.im*(*sd).c3.c3.im;

      ns+=
         (*sd).c4.c1.re*(*sd).c4.c1.re+(*sd).c4.c1.im*(*sd).c4.c1.im+
         (*sd).c4.c2.re*(*sd).c4.c2.re+(*sd).c4.c2.im*(*sd).c4.c2.im+
         (*sd).c4.c3.re*(*sd).c4.c3.re+(*sd).c4.c3.im*(*sd).c4.c3.im;

      if (ns>nrm)
         nrm=ns;
   }

   return nrm;
}

#endif

float unorm(int vol,int icom,spinor *s)
{
   float mxn,nrm;

   mxn=loc_unorm(vol,s);

   if ((NPROC>1)&&(icom==1))
   {
      nrm=mxn;
      MPI_Allreduce(&nrm,&mxn,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
   }

   return (float)(sqrt((double)(mxn)));
}


double unorm_dble(int vol,int icom,spinor_dble *sd)
{
   double mxn,nrm;

   mxn=loc_unorm_dble(vol,sd);

   if ((NPROC>1)&&(icom==1))
   {
      nrm=mxn;
      MPI_Allreduce(&nrm,&mxn,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
   }

   return sqrt(mxn);
}
