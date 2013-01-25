
/*******************************************************************************
*
* File time2.c
*
* Copyright (C) 2005, 2008, 2009, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the SU(3) x SU(3)-vector multiplications (double-precision
* programs)
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
static su3_dble u[4] ALIGNED16;
static su3_vector_dble s[4],r[4],t[4] ALIGNED16;

#if (defined x64)
#include "sse2.h"

#define _su3_fast_multiply(r,u,s) \
   _sse_load_dble(s); \
   _sse_su3_multiply_dble(u); \
   _sse_store_up_dble(r)

#define _su3_fast_inverse_multiply(r,u,s) \
   _sse_load_dble(s); \
   _sse_su3_inverse_multiply_dble(u); \
   _sse_store_up_dble(r)


static void fast_multiply(su3_dble *ua,su3_vector_dble *sa,
                          su3_vector_dble *ra)
{
   _su3_fast_multiply((*(ra  )),(*(ua  )),(*(sa  )));
   _su3_fast_multiply((*(ra+1)),(*(ua+1)),(*(sa+1)));
   _su3_fast_multiply((*(ra+2)),(*(ua+2)),(*(sa+2)));
   _su3_fast_multiply((*(ra+3)),(*(ua+3)),(*(sa+3)));
}


static void fast_inverse_multiply(su3_dble *ua,su3_vector_dble *sa,
                                  su3_vector_dble *ra)
{
   _su3_fast_inverse_multiply((*(ra  )),(*(ua  )),(*(sa  )));
   _su3_fast_inverse_multiply((*(ra+1)),(*(ua+1)),(*(sa+1)));
   _su3_fast_inverse_multiply((*(ra+2)),(*(ua+2)),(*(sa+2)));
   _su3_fast_inverse_multiply((*(ra+3)),(*(ua+3)),(*(sa+3)));
}

#endif

static void slow_multiply(su3_dble *ua,su3_vector_dble *sa,
                          su3_vector_dble *ra)
{
   _su3_multiply((*(ra  )),(*(ua  )),(*(sa  )));
   _su3_multiply((*(ra+1)),(*(ua+1)),(*(sa+1)));
   _su3_multiply((*(ra+2)),(*(ua+2)),(*(sa+2)));
   _su3_multiply((*(ra+3)),(*(ua+3)),(*(sa+3)));
}


int main(void)
{
   int k,n,count;
   double t1,t2,dt;
   double delta,diff,norm;

   printf("\n");
   printf("Time per double-precision SU(3) x SU(3)-vector multiplication\n");
   printf("-------------------------------------------------------------\n\n");

#if (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n");
#endif

   printf("Measurement made with all data in cache\n\n");
   
   rlxd_init(1,123456);

   for (k=0;k<4;k++)
      random_su3_dble(u+k);

   gauss_dble((double*)(s),24);
   gauss_dble((double*)(r),24);
   gauss_dble((double*)(t),24);      
   
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

   dt*=1.0e6/(double)(2*n);

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
   
   dt*=1.0e6/(double)(2*n);

   printf("The time per v=U^dag*w is %4.3f micro sec",dt);
   printf(" [%d Mflops]\n",(int)(66.0/dt));

   fast_multiply(u,s,r);
   fast_inverse_multiply(u,r,t);   
   delta=0.0;

   for (k=0;k<4;k++)
   {
      _vector_sub_assign(t[k],s[k]);
      diff=_vector_prod_re(t[k],t[k]);
      norm=_vector_prod_re(s[k],s[k]);
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
      for (count=0;count<n;count++)
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

   for (k=0;k<4;k++)
   {
      _vector_sub_assign(r[k],t[k]);
      diff=_vector_prod_re(r[k],r[k]);
      norm=_vector_prod_re(s[k],s[k]);
      diff=sqrt(diff/norm);
      if (diff>delta)
         delta=diff;
   }

   printf("||v_SSE-v_FPU||<= %.1e*||w||\n",delta);

#endif

   slow_multiply(u,s,t);    
   _su3_inverse_multiply((*(r  )),(*(u  )),(*(t  )));
   _su3_inverse_multiply((*(r+1)),(*(u+1)),(*(t+1)));
   _su3_inverse_multiply((*(r+2)),(*(u+2)),(*(t+2)));
   _su3_inverse_multiply((*(r+3)),(*(u+3)),(*(t+3)));

   delta=0.0;

   for (k=0;k<4;k++)
   {
      _vector_sub_assign(r[k],s[k]);
      diff=_vector_prod_re(r[k],r[k]);
      norm=_vector_prod_re(s[k],s[k]);
      diff=sqrt(diff/norm);
      if (diff>delta)
         delta=diff;
   }

   printf("||w-U^dag*U*w||<= %.1e*||w||\n\n",delta);

   exit(0);
}
