
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2005, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of mul_pauli()
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include "random.h"
#include "linalg.h"
#include "sw_term.h"

typedef union
{
   weyl w;
   float r[12];
} spin_t;

static pauli mp1,mp2 ALIGNED16;
static spin_t s1,s2,r1,r2 ALIGNED16;


int main(void)
{
   int n,count;
   float mu1,mu2;
   double t1,t2,dt;

   printf("\n");
   printf("Timing of mul_pauli()\n");
   printf("---------------------\n\n");

#if (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n\n");
#endif

   printf("Measurement made with all data in cache\n\n");
   
   rlxs_init(0,23456);
   ranlxs(mp1.u,36);
   ranlxs(mp2.u,36);
   ranlxs(s1.r,12);
   ranlxs(s2.r,12);
   mu1=0.1234f;
   mu2=0.5678f;

   n=(int)(1.0e6);
   dt=0.0;

   while (dt<2.0)
   {
      t1=(double)clock();
      for (count=0;count<n;count++)
      {
         mul_pauli(mu1,&mp1,&(s1.w),&(r1.w));
         mul_pauli(mu2,&mp2,&(s2.w),&(r2.w));
      }
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }

   dt*=(1.0e6/(double)(n));

   printf("Time per call of mul_pauli():\n");
   printf("%.4f usec (%d Mflops [%d bit arithmetic])\n\n",
          dt,(int)(276.0/dt),(int)(sizeof(spinor)/3));

   exit(0);
}
