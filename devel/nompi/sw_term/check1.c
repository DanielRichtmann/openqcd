
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2005, 2009, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of mul_pauli()
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "random.h"
#include "utils.h"
#include "random.h"
#include "linalg.h"
#include "sw_term.h"

typedef union
{
   weyl w;
   complex c[6];
   float r[12];
} spin_t;

typedef union
{
   complex c[36];
   float r[72];
} mat_t;

static pauli mp ALIGNED16;
static spin_t s1,s2,r1,r2 ALIGNED16;
static mat_t mv ALIGNED16;


static void cpvec(int n,complex *s,complex *r)
{
   int i;

   for (i=0;i<n;i++)
      r[i]=s[i];
}


static int diffvec(int n,complex *s,complex *r)
{
   int i;

   for (i=0;i<n;i++)
   {
      if ((s[i].re!=r[i].re)||(s[i].im!=r[i].im))
         return 1;
   }

   return 0;
}


int main(void)
{
   int i,j,k;
   float mu,d,dmax;

   printf("\n");
   printf("Check of mul_pauli()\n");
   printf("--------------------\n\n");

#if (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n\n");
#endif

   rlxs_init(0,3898);
   gauss(s1.r,12);
   gauss(mv.r,72);
   mu=0.1234f;

   for (i=0;i<6;i++)
   {
      for (j=i;j<6;j++)
      {
         if (j>i)
         {
            mv.c[6*i+j].re= mv.c[6*j+i].re;
            mv.c[6*i+j].im=-mv.c[6*j+i].im;
         }
         else
            mv.c[6*i+j].im=0.0f;
      }
   }

   k=6;
   
   for (i=0;i<6;i++)
   {
      mp.u[i]=mv.c[6*i+i].re;

      for (j=i+1;j<6;j++)
      {
         mp.u[k]=mv.c[6*i+j].re;
         k+=1;
         mp.u[k]=mv.c[6*i+j].im;
         k+=1;
      }
   }

   cpvec(6,s1.c,s2.c);
   mul_pauli(mu,&mp,&(s1.w),&(r1.w));

   error(diffvec(6,s1.c,s2.c),1,"main [check1.c]",
         "mul_pauli() modifies the source spinor"); 
   
   cmat_vec(6,mv.c,s2.c,r2.c);

   for (i=0;i<6;i++)
   {
      r2.c[i].re-=mu*s2.c[i].im;
      r2.c[i].im+=mu*s2.c[i].re;
   }
   
   printf("r1: result, r2: expected result\n\n");
   
   for (i=0;i<2;i++)
   {
      for (j=0;j<3;j++)
      {
         k=3*i+j;
         printf("r1.c%d.c%d=(% .7e,% .7e)\n",i+1,j+1,r1.c[k].re,r1.c[k].im);
         printf("r2.c%d.c%d=(% .7e,% .7e)\n",i+1,j+1,r2.c[k].re,r2.c[k].im);
         printf("\n");
      }
   }

   dmax=0.0f;

   for (i=0;i<12;i++)
   {
      d=(float)(fabs((double)(r1.r[i]-r2.r[i])));
      if (d>dmax)
         dmax=d;
   }

   printf("Maximal absolute deviation = %.1e\n",dmax);
   
   mul_pauli(mu,&mp,&(s1.w),&(s1.w));
   error(diffvec(6,s1.c,r1.c),1,"main [check1.c]",
         "mul_pauli() is incorrect when r=s");    
   printf("Works correctly if input and output spinors coincide\n\n");
   
   exit(0);
}
