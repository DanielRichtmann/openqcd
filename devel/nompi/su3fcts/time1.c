
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2005, 2008, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the SU(3) x SU(3)-vector multiplication (single-precision programs)
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "su3.h"
#include "random.h"
#include "su3fcts.h"

static su3 u[4] ALIGNED16;
static su3_vector s[8],r[8],t[8] ALIGNED16;

#if (defined x64)
#include "sse2.h"

#define _su3_fast_multiply(r1,r2,u,s1,s2) \
   _sse_pair_load(s1,s2); \
   _sse_su3_multiply(u); \
   _sse_pair_store_up(r1,r2)

#define _su3_fast_inverse_multiply(r1,r2,u,s1,s2) \
   _sse_pair_load(s1,s2); \
   _sse_su3_inverse_multiply(u); \
   _sse_pair_store_up(r1,r2)


static void fast_multiply(su3 *ua,su3_vector *sa,su3_vector *ra)
{
   _su3_fast_multiply((*(ra  )),(*(ra+1)),(*(ua  )),(*(sa  )),(*(sa+1)));
   _su3_fast_multiply((*(ra+2)),(*(ra+3)),(*(ua+1)),(*(sa+2)),(*(sa+3)));
   _su3_fast_multiply((*(ra+4)),(*(ra+5)),(*(ua+2)),(*(sa+4)),(*(sa+5)));
   _su3_fast_multiply((*(ra+6)),(*(ra+7)),(*(ua+3)),(*(sa+6)),(*(sa+7)));
}


static void fast_inverse_multiply(su3 *ua,su3_vector *sa,su3_vector *ra)
{
   _su3_fast_inverse_multiply((*(ra  )),(*(ra+1)),(*(ua  )),(*(sa  )),(*(sa+1)));
   _su3_fast_inverse_multiply((*(ra+2)),(*(ra+3)),(*(ua+1)),(*(sa+2)),(*(sa+3)));
   _su3_fast_inverse_multiply((*(ra+4)),(*(ra+5)),(*(ua+2)),(*(sa+4)),(*(sa+5)));
   _su3_fast_inverse_multiply((*(ra+6)),(*(ra+7)),(*(ua+3)),(*(sa+6)),(*(sa+7)));
}

#endif

static void slow_multiply(su3 *ua,su3_vector *sa,su3_vector *ra)
{
   _su3_multiply((*(ra  )),(*(ua  )),(*(sa  )));
   _su3_multiply((*(ra+1)),(*(ua  )),(*(sa+1)));
   _su3_multiply((*(ra+2)),(*(ua+1)),(*(sa+2)));
   _su3_multiply((*(ra+3)),(*(ua+1)),(*(sa+3)));
   _su3_multiply((*(ra+4)),(*(ua+2)),(*(sa+4)));
   _su3_multiply((*(ra+5)),(*(ua+2)),(*(sa+5)));
   _su3_multiply((*(ra+6)),(*(ua+3)),(*(sa+6)));
   _su3_multiply((*(ra+7)),(*(ua+3)),(*(sa+7)));
}


int main(void)
{
   int k,n,count;
   double t1,t2,dt;
   double delta,diff,norm;

   printf("\n");
   printf("Time per single-precision SU(3) x SU(3)-vector multiplication\n");
   printf("-------------------------------------------------------------\n\n");

#if (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n");
#endif

   printf("Measurement made with all data in cache\n\n");   
   
   rlxs_init(0,123456);

   for (k=0;k<4;k++)
      random_su3(u+k);

   gauss((float*)(s),48);
   gauss((float*)(r),48);
   gauss((float*)(t),48);      

#if (defined x64)

   n=(int)(1.0e6);
   dt=0.0;

   while (dt<2.0)
   { 
      t1=(double)clock();
      for (count=0;count<n;count++)
         fast_multiply(u,s,r);
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }
   
   dt*=1.0e6/(double)(4*n);

   printf("The time per v=U*w is     %4.3f micro sec",dt);
   printf(" [%d Mflops]\n",(int)(66.0/dt));

   n=(int)(1.0e6);
   dt=0.0;

   while (dt<2.0)
   {   
      t1=(double)clock();
      for (count=0;count<n;count++)
         fast_inverse_multiply(u,s,r);
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }

   dt*=1.0e6/(double)(4*n);
   
   printf("The time per v=U^dag*w is %4.3f micro sec",dt);
   printf(" [%d Mflops]\n",(int)(66.0/dt));

   fast_multiply(u,s,r);
   fast_inverse_multiply(u,r,t);
   delta=0.0;

   for (k=0;k<8;k++)
   {
      _vector_sub_assign(t[k],s[k]);
      diff=(double)(_vector_prod_re(t[k],t[k]));
      norm=(double)(_vector_prod_re(s[k],s[k]));
      diff=sqrt(diff/norm);
      if (diff>delta)
         delta=diff;
   }

   printf("||w-U^dag*U*w||<= %.1e*||w||\n\n",delta);

#endif

   n=(int)(1.0e6);
   dt=0.0;

   while (dt<2.0)
   {
      t1=(double)clock();
      for (count=0;count<(n/2);count++)
         slow_multiply(u,s,t);
      t2=(double)clock();
      dt=(t2-t1)/(double)(CLOCKS_PER_SEC);
      n*=2;
   }
   
   dt*=1.0e6/(double)(2*n);

   printf("Using x87 FPU instructions:\n");
   printf("The time per v=U*w is     %4.3f micro sec",dt);
   printf(" [%d Mflops]\n",(int)(66.0/dt));

#if (defined x64)

   fast_multiply(u,s,r);
   slow_multiply(u,s,t);
   delta=0.0;

   for (k=0;k<8;k++)
   {
      _vector_sub_assign(r[k],t[k]);
      diff=(double)(_vector_prod_re(r[k],r[k]));
      norm=(double)(_vector_prod_re(s[k],s[k]));
      diff=sqrt(diff/norm);
      if (diff>delta)
         delta=diff;
   }

   printf("||v_SSE-v_FPU||<= %.1e*||w||\n",delta);

#endif

   slow_multiply(u,s,t);   
   _su3_inverse_multiply((*(r  )),(*(u  )),(*(t  )));
   _su3_inverse_multiply((*(r+1)),(*(u  )),(*(t+1)));
   _su3_inverse_multiply((*(r+2)),(*(u+1)),(*(t+2)));
   _su3_inverse_multiply((*(r+3)),(*(u+1)),(*(t+3)));
   _su3_inverse_multiply((*(r+4)),(*(u+2)),(*(t+4)));
   _su3_inverse_multiply((*(r+5)),(*(u+2)),(*(t+5)));
   _su3_inverse_multiply((*(r+6)),(*(u+3)),(*(t+6)));
   _su3_inverse_multiply((*(r+7)),(*(u+3)),(*(t+7)));

   delta=0.0;

   for (k=0;k<8;k++)
   {
      _vector_sub_assign(r[k],s[k]);
      diff=(double)(_vector_prod_re(r[k],r[k]));
      norm=(double)(_vector_prod_re(s[k],s[k]));
      diff=sqrt(diff/norm);
      if (diff>delta)
         delta=diff;
   }

   printf("||w-U^dag*U*w||<= %.1e*||w||\n\n",delta);

   exit(0);
}

