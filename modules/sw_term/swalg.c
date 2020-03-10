
/*******************************************************************************
*
* File swalg.c
*
* Copyright (C) 2018 Antonio Rago, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Linear algebra functions for matrices of type pauli_dble.
*
*   void pauli2weyl(pauli_dble *A,weyl_dble *w)
*     Assigns the column vectors of the matrix A to the Weyl spinors
*     w[0],..,w[5].
*
*   void weyl2pauli(weyl_dble *w,pauli_dble *A)
*     Assigns the Weyl spinors w[0],..,w[5] to the columns of the matrix A[0].
*
*   void prod_pauli_mat(pauli_dble *A,weyl_dble *w,weyl_dble *v)
*     Assigns the products A[0]*w[k] to v[k] for k=0,..,5. It is permissible
*     to set v=w.
*
*   void add_pauli_mat(pauli_dble *A,pauli_dble *B,pauli_dble *C)
*     Assigns the sum A[0]+B[0] to C[0]. It is permissible to set C=A
*     or C=B.
*
*   void lc3_pauli_mat(double *c,pauli_dble *A1,pauli_dble *A2,pauli_dble *B)
*     Assigns the linear combination c[0]+c[1]*A1[0]+c[2]*A2[0] to B[0]. It
*     is permissible to set B=A1 or B=A2.
*
*   double tr0_pauli_mat(pauli_dble *A)
*     Returns the trace of the matrix A[0].
*
*   double tr1_pauli_mat(pauli_dble *A,pauli_dble *B)
*     Returns the trace of the product of the matrices A[0] and B[0].
*
* The programs in this module do not perform any communications and can be
* locally called. If SSE (AVX) instructions are used, the Pauli matrices and
* Weyl spinors must be aligned to a 16 (32) byte boundary.
*
*******************************************************************************/

#define SWALG_C

#include <stdlib.h>
#include <stdio.h>
#include "sw_term.h"

#if (defined x64)

typedef struct
{
   double u[2];
} vec2;

typedef union
{
   pauli_dble A;
   vec2 v[18];
}
vec_t;

#endif

void pauli2weyl(pauli_dble *A,weyl_dble *w)
{
   double *u;

   u=(*A).u;

   w[0].c1.c1.re=u[0];
   w[0].c1.c1.im=0.0;

   w[1].c1.c2.re=u[1];
   w[1].c1.c2.im=0.0;

   w[2].c1.c3.re=u[2];
   w[2].c1.c3.im=0.0;

   w[3].c2.c1.re=u[3];
   w[3].c2.c1.im=0.0;

   w[4].c2.c2.re=u[4];
   w[4].c2.c2.im=0.0;

   w[5].c2.c3.re=u[5];
   w[5].c2.c3.im=0.0;

   w[1].c1.c1.re=u[6];
   w[1].c1.c1.im=u[7];

   w[0].c1.c2.re=u[6];
   w[0].c1.c2.im=-u[7];

   w[2].c1.c1.re=u[8];
   w[2].c1.c1.im=u[9];

   w[0].c1.c3.re=u[8];
   w[0].c1.c3.im=-u[9];

   w[3].c1.c1.re=u[10];
   w[3].c1.c1.im=u[11];

   w[0].c2.c1.re=u[10];
   w[0].c2.c1.im=-u[11];

   w[4].c1.c1.re=u[12];
   w[4].c1.c1.im=u[13];

   w[0].c2.c2.re=u[12];
   w[0].c2.c2.im=-u[13];

   w[5].c1.c1.re=u[14];
   w[5].c1.c1.im=u[15];

   w[0].c2.c3.re=u[14];
   w[0].c2.c3.im=-u[15];

   w[2].c1.c2.re=u[16];
   w[2].c1.c2.im=u[17];

   w[1].c1.c3.re=u[16];
   w[1].c1.c3.im=-u[17];

   w[3].c1.c2.re=u[18];
   w[3].c1.c2.im=u[19];

   w[1].c2.c1.re=u[18];
   w[1].c2.c1.im=-u[19];

   w[4].c1.c2.re=u[20];
   w[4].c1.c2.im=u[21];

   w[1].c2.c2.re=u[20];
   w[1].c2.c2.im=-u[21];

   w[5].c1.c2.re=u[22];
   w[5].c1.c2.im=u[23];

   w[1].c2.c3.re=u[22];
   w[1].c2.c3.im=-u[23];

   w[3].c1.c3.re=u[24];
   w[3].c1.c3.im=u[25];

   w[2].c2.c1.re=u[24];
   w[2].c2.c1.im=-u[25];

   w[4].c1.c3.re=u[26];
   w[4].c1.c3.im=u[27];

   w[2].c2.c2.re=u[26];
   w[2].c2.c2.im=-u[27];

   w[5].c1.c3.re=u[28];
   w[5].c1.c3.im=u[29];

   w[2].c2.c3.re=u[28];
   w[2].c2.c3.im=-u[29];

   w[4].c2.c1.re=u[30];
   w[4].c2.c1.im=u[31];

   w[3].c2.c2.re=u[30];
   w[3].c2.c2.im=-u[31];

   w[5].c2.c1.re=u[32];
   w[5].c2.c1.im=u[33];

   w[3].c2.c3.re=u[32];
   w[3].c2.c3.im=-u[33];

   w[5].c2.c2.re=u[34];
   w[5].c2.c2.im=u[35];

   w[4].c2.c3.re=u[34];
   w[4].c2.c3.im=-u[35];
}


void weyl2pauli(weyl_dble *w,pauli_dble *A)
{
   double *u;

   u=(*A).u;

   u[0]=w[0].c1.c1.re;

   u[6]=w[1].c1.c1.re;
   u[7]=w[1].c1.c1.im;
   u[1]=w[1].c1.c2.re;

   u[8]=w[2].c1.c1.re;
   u[9]=w[2].c1.c1.im;
   u[16]=w[2].c1.c2.re;
   u[17]=w[2].c1.c2.im;
   u[2]=w[2].c1.c3.re;

   u[10]=w[3].c1.c1.re;
   u[11]=w[3].c1.c1.im;
   u[18]=w[3].c1.c2.re;
   u[19]=w[3].c1.c2.im;
   u[24]=w[3].c1.c3.re;
   u[25]=w[3].c1.c3.im;
   u[3]=w[3].c2.c1.re;

   u[12]=w[4].c1.c1.re;
   u[13]=w[4].c1.c1.im;
   u[20]=w[4].c1.c2.re;
   u[21]=w[4].c1.c2.im;
   u[26]=w[4].c1.c3.re;
   u[27]=w[4].c1.c3.im;
   u[30]=w[4].c2.c1.re;
   u[31]=w[4].c2.c1.im;
   u[4]=w[4].c2.c2.re;

   u[14]=w[5].c1.c1.re;
   u[15]=w[5].c1.c1.im;
   u[22]=w[5].c1.c2.re;
   u[23]=w[5].c1.c2.im;
   u[28]=w[5].c1.c3.re;
   u[29]=w[5].c1.c3.im;
   u[32]=w[5].c2.c1.re;
   u[33]=w[5].c2.c1.im;
   u[34]=w[5].c2.c2.re;
   u[35]=w[5].c2.c2.im;
   u[5]=w[5].c2.c3.re;
}


void prod_pauli_mat(pauli_dble *A,weyl_dble *w,weyl_dble *v)
{
   weyl_dble *wm;

   wm=w+6;

   for (;w<wm;w++)
   {
      mul_pauli_dble(0.0,A,w,v);
      v+=1;
   }
}


void add_pauli_mat(pauli_dble *A,pauli_dble *B,pauli_dble *C)
{
   double *u,*v,*w,*um;

   u=A[0].u;
   v=B[0].u;
   w=C[0].u;
   um=u+36;

   for (;u<um;u++)
   {
      w[0]=u[0]+v[0];
      v+=1;
      w+=1;
   }
}


double tr0_pauli_mat(pauli_dble *A)
{
   double *u;

   u=A[0].u;

   return u[0]+u[1]+u[2]+u[3]+u[4]+u[5];
}

#if (defined x64)

double tr1_pauli_mat(pauli_dble *A,pauli_dble *B)
{
   double tr;
   vec_t *vA,*vB;
   vec2 *u,*v;

   vA=(vec_t*)(A);
   vB=(vec_t*)(B);
   u=(vec2*)(vA);
   v=(vec2*)(vB);

   __asm__ __volatile__ ("movapd %0, %%xmm0 \n\t"
                         "movapd %1, %%xmm1 \n\t"
                         "movapd %2, %%xmm2 \n\t"
                         "movapd %3, %%xmm3 \n\t"
                         "movapd %4, %%xmm4 \n\t"
                         "movapd %5, %%xmm5"
                         :
                         :
                         "m" (u[0]),
                         "m" (u[1]),
                         "m" (u[2]),
                         "m" (u[3]),
                         "m" (u[4]),
                         "m" (u[5])
                         :
                         "xmm0", "xmm1", "xmm2",
                         "xmm3", "xmm4", "xmm5");

   __asm__ __volatile__ ("mulpd %0, %%xmm0 \n\t"
                         "mulpd %1, %%xmm1 \n\t"
                         "mulpd %2, %%xmm2 \n\t"
                         "mulpd %3, %%xmm3 \n\t"
                         "mulpd %4, %%xmm4 \n\t"
                         "mulpd %5, %%xmm5"
                         :
                         :
                         "m" (v[0]),
                         "m" (v[1]),
                         "m" (v[2]),
                         "m" (v[3]),
                         "m" (v[4]),
                         "m" (v[5])
                         :
                         "xmm0", "xmm1", "xmm2",
                         "xmm3", "xmm4", "xmm5");

   __asm__ __volatile__ ("movapd %0, %%xmm6 \n\t"
                         "movapd %1, %%xmm7 \n\t"
                         "movapd %2, %%xmm8 \n\t"
                         "movapd %3, %%xmm9 \n\t"
                         "movapd %4, %%xmm10 \n\t"
                         "movapd %5, %%xmm11"
                         :
                         :
                         "m" (u[6]),
                         "m" (u[7]),
                         "m" (u[8]),
                         "m" (u[9]),
                         "m" (u[10]),
                         "m" (u[11])
                         :
                         "xmm6", "xmm7", "xmm8",
                         "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("mulpd %0, %%xmm6 \n\t"
                         "mulpd %1, %%xmm7 \n\t"
                         "mulpd %2, %%xmm8 \n\t"
                         "mulpd %3, %%xmm9 \n\t"
                         "mulpd %4, %%xmm10 \n\t"
                         "mulpd %5, %%xmm11"
                         :
                         :
                         "m" (v[6]),
                         "m" (v[7]),
                         "m" (v[8]),
                         "m" (v[9]),
                         "m" (v[10]),
                         "m" (v[11])
                         :
                         "xmm6", "xmm7", "xmm8",
                         "xmm9", "xmm10", "xmm11");

   __asm__ __volatile__ ("addpd %%xmm1, %%xmm0 \n\t"
                         "addpd %%xmm4, %%xmm3 \n\t"
                         "addpd %%xmm6, %%xmm5 \n\t"
                         "addpd %%xmm8, %%xmm7 \n\t"
                         "addpd %%xmm10, %%xmm9"
                         :
                         :
                         :
                         "xmm0", "xmm3", "xmm5",
                         "xmm7", "xmm9");

   __asm__ __volatile__ ("movapd %0, %%xmm12 \n\t"
                         "movapd %1, %%xmm13 \n\t"
                         "movapd %2, %%xmm14 \n\t"
                         "movapd %3, %%xmm4 \n\t"
                         "movapd %4, %%xmm6 \n\t"
                         "movapd %5, %%xmm8"
                         :
                         :
                         "m" (u[12]),
                         "m" (u[13]),
                         "m" (u[14]),
                         "m" (u[15]),
                         "m" (u[16]),
                         "m" (u[17])
                         :
                         "xmm4", "xmm6", "xmm8",
                         "xmm12", "xmm13", "xmm14");

   __asm__ __volatile__ ("mulpd %0, %%xmm12 \n\t"
                         "mulpd %1, %%xmm13 \n\t"
                         "mulpd %2, %%xmm14 \n\t"
                         "mulpd %3, %%xmm4 \n\t"
                         "mulpd %4, %%xmm6 \n\t"
                         "mulpd %5, %%xmm8"
                         :
                         :
                         "m" (v[12]),
                         "m" (v[13]),
                         "m" (v[14]),
                         "m" (v[15]),
                         "m" (v[16]),
                         "m" (v[17])
                         :
                         "xmm4", "xmm6", "xmm8",
                         "xmm12", "xmm13", "xmm14");

   __asm__ __volatile__ ("addpd %%xmm2, %%xmm0 \n\t"
                         "addpd %%xmm11, %%xmm3 \n\t"
                         "addpd %%xmm12, %%xmm5 \n\t"
                         "addpd %%xmm13, %%xmm7 \n\t"
                         "addpd %%xmm14, %%xmm9 \n\t"
                         "addpd %%xmm4, %%xmm3 \n\t"
                         "addpd %%xmm6, %%xmm5 \n\t"
                         "addpd %%xmm8, %%xmm7 \n\t"
                         "addpd %%xmm9, %%xmm3 \n\t"
                         "addpd %%xmm7, %%xmm5 \n\t"
                         "addpd %%xmm5, %%xmm3"
                         :
                         :
                         :
                         "xmm0", "xmm3", "xmm5",
                         "xmm7", "xmm9");

   __asm__ __volatile__ ("addpd %%xmm3, %%xmm0 \n\t"
                         "addpd %%xmm3, %%xmm0 \n\t"
                         "haddpd %%xmm0, %%xmm0 \n\t"
                         "movsd %%xmm0, %0"
                         :
                         "=m" (tr)
                         :
                         :
                         "xmm0");

   return tr;
}


void lc3_pauli_mat(double *c,pauli_dble *A1,pauli_dble *A2,pauli_dble *B)
{
   vec_t *vA1,*vA2,*vB;
   vec2 *u,*v,*w;

   vA1=(vec_t*)(A1);
   vA2=(vec_t*)(A2);
   vB=(vec_t*)(B);
   u=(vec2*)(vA1);
   v=(vec2*)(vA2);
   w=(vec2*)(vB);

   __asm__ __volatile__ ("movddup %0, %%xmm0 \n\t"
                         "movddup %1, %%xmm1 \n\t"
                         "movddup %2, %%xmm2"
                         :
                         :
                         "m" (c[0]),
                         "m" (c[1]),
                         "m" (c[2])
                         :
                         "xmm0", "xmm1", "xmm2");

   __asm__ __volatile__ ("movapd %0, %%xmm3 \n\t"
                         "movapd %1, %%xmm4 \n\t"
                         "movapd %2, %%xmm5 \n\t"
                         "movapd %3, %%xmm6 \n\t"
                         "movapd %4, %%xmm7 \n\t"
                         "movapd %5, %%xmm8 \n\t"
                         "movapd %6, %%xmm9 \n\t"
                         "movapd %7, %%xmm10"
                         :
                         :
                         "m" (u[0]),
                         "m" (v[0]),
                         "m" (u[1]),
                         "m" (v[1]),
                         "m" (u[2]),
                         "m" (v[2]),
                         "m" (u[3]),
                         "m" (v[3])
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm7", "xmm8", "xmm9", "xmm10");

   __asm__ __volatile__ ("mulpd %%xmm1, %%xmm3 \n\t"
                         "mulpd %%xmm2, %%xmm4 \n\t"
                         "mulpd %%xmm1, %%xmm5 \n\t"
                         "mulpd %%xmm2, %%xmm6 \n\t"
                         "mulpd %%xmm1, %%xmm7 \n\t"
                         "mulpd %%xmm2, %%xmm8 \n\t"
                         "mulpd %%xmm1, %%xmm9 \n\t"
                         "mulpd %%xmm2, %%xmm10"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm7", "xmm8", "xmm9", "xmm10");

   __asm__ __volatile__ ("addpd %%xmm4, %%xmm3 \n\t"
                         "addpd %%xmm6, %%xmm5 \n\t"
                         "addpd %%xmm8, %%xmm7 \n\t"
                         "addpd %%xmm10, %%xmm9 \n\t"
                         "addpd %%xmm0, %%xmm3 \n\t"
                         "addpd %%xmm0, %%xmm5 \n\t"
                         "addpd %%xmm0, %%xmm7 \n\t"
                         "movapd %%xmm3, %0 \n\t"
                         "movapd %%xmm5, %1 \n\t"
                         "movapd %%xmm7, %2 \n\t"
                         "movapd %%xmm9, %3"
                         :
                         "=m" (w[0]),
                         "=m" (w[1]),
                         "=m" (w[2]),
                         "=m" (w[3])
                         :
                         :
                         "xmm3", "xmm5", "xmm7", "xmm9");

   __asm__ __volatile__ ("movapd %0, %%xmm11 \n\t"
                         "movapd %1, %%xmm12 \n\t"
                         "movapd %2, %%xmm13 \n\t"
                         "movapd %3, %%xmm14 \n\t"
                         "movapd %4, %%xmm3 \n\t"
                         "movapd %5, %%xmm4 \n\t"
                         "movapd %6, %%xmm5 \n\t"
                         "movapd %7, %%xmm6"
                         :
                         :
                         "m" (u[4]),
                         "m" (v[4]),
                         "m" (u[5]),
                         "m" (v[5]),
                         "m" (u[6]),
                         "m" (v[6]),
                         "m" (u[7]),
                         "m" (v[7])
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm11", "xmm12", "xmm13", "xmm14");

   __asm__ __volatile__ ("mulpd %%xmm1, %%xmm11 \n\t"
                         "mulpd %%xmm2, %%xmm12 \n\t"
                         "mulpd %%xmm1, %%xmm13 \n\t"
                         "mulpd %%xmm2, %%xmm14 \n\t"
                         "mulpd %%xmm1, %%xmm3 \n\t"
                         "mulpd %%xmm2, %%xmm4 \n\t"
                         "mulpd %%xmm1, %%xmm5 \n\t"
                         "mulpd %%xmm2, %%xmm6"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm11", "xmm12", "xmm13", "xmm14");

   __asm__ __volatile__ ("addpd %%xmm12, %%xmm11 \n\t"
                         "addpd %%xmm14, %%xmm13 \n\t"
                         "addpd %%xmm4, %%xmm3 \n\t"
                         "addpd %%xmm6, %%xmm5 \n\t"
                         "movapd %%xmm11, %0 \n\t"
                         "movapd %%xmm13, %1 \n\t"
                         "movapd %%xmm3, %2 \n\t"
                         "movapd %%xmm5, %3"
                         :
                         "=m" (w[4]),
                         "=m" (w[5]),
                         "=m" (w[6]),
                         "=m" (w[7])
                         :
                         :
                         "xmm3", "xmm5", "xmm11", "xmm13");

   __asm__ __volatile__ ("movapd %0, %%xmm7 \n\t"
                         "movapd %1, %%xmm8 \n\t"
                         "movapd %2, %%xmm9 \n\t"
                         "movapd %3, %%xmm10 \n\t"
                         "movapd %4, %%xmm3 \n\t"
                         "movapd %5, %%xmm4 \n\t"
                         "movapd %6, %%xmm5 \n\t"
                         "movapd %7, %%xmm6"
                         :
                         :
                         "m" (u[8]),
                         "m" (v[8]),
                         "m" (u[9]),
                         "m" (v[9]),
                         "m" (u[10]),
                         "m" (v[10]),
                         "m" (u[11]),
                         "m" (v[11])
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm7", "xmm8", "xmm9", "xmm10");

   __asm__ __volatile__ ("mulpd %%xmm1, %%xmm7 \n\t"
                         "mulpd %%xmm2, %%xmm8 \n\t"
                         "mulpd %%xmm1, %%xmm9 \n\t"
                         "mulpd %%xmm2, %%xmm10 \n\t"
                         "mulpd %%xmm1, %%xmm3 \n\t"
                         "mulpd %%xmm2, %%xmm4 \n\t"
                         "mulpd %%xmm1, %%xmm5 \n\t"
                         "mulpd %%xmm2, %%xmm6"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm7", "xmm8", "xmm9", "xmm10");

   __asm__ __volatile__ ("addpd %%xmm8, %%xmm7 \n\t"
                         "addpd %%xmm10, %%xmm9 \n\t"
                         "addpd %%xmm4, %%xmm3 \n\t"
                         "addpd %%xmm6, %%xmm5 \n\t"
                         "movapd %%xmm7, %0 \n\t"
                         "movapd %%xmm9, %1 \n\t"
                         "movapd %%xmm3, %2 \n\t"
                         "movapd %%xmm5, %3"
                         :
                         "=m" (w[8]),
                         "=m" (w[9]),
                         "=m" (w[10]),
                         "=m" (w[11])
                         :
                         :
                         "xmm3", "xmm5", "xmm7", "xmm9");

   __asm__ __volatile__ ("movapd %0, %%xmm11 \n\t"
                         "movapd %1, %%xmm12 \n\t"
                         "movapd %2, %%xmm13 \n\t"
                         "movapd %3, %%xmm14 \n\t"
                         "movapd %4, %%xmm3 \n\t"
                         "movapd %5, %%xmm4 \n\t"
                         "movapd %6, %%xmm5 \n\t"
                         "movapd %7, %%xmm6"
                         :
                         :
                         "m" (u[12]),
                         "m" (v[12]),
                         "m" (u[13]),
                         "m" (v[13]),
                         "m" (u[14]),
                         "m" (v[14]),
                         "m" (u[15]),
                         "m" (v[15])
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm11", "xmm12", "xmm13", "xmm14");

   __asm__ __volatile__ ("movapd %0, %%xmm7 \n\t"
                         "movapd %1, %%xmm8 \n\t"
                         "movapd %2, %%xmm9 \n\t"
                         "movapd %3, %%xmm10"
                         :
                         :
                         "m" (u[16]),
                         "m" (v[16]),
                         "m" (u[17]),
                         "m" (v[17])
                         :
                         "xmm7", "xmm8", "xmm9", "xmm10");

   __asm__ __volatile__ ("mulpd %%xmm1, %%xmm11 \n\t"
                         "mulpd %%xmm2, %%xmm12 \n\t"
                         "mulpd %%xmm1, %%xmm13 \n\t"
                         "mulpd %%xmm2, %%xmm14 \n\t"
                         "mulpd %%xmm1, %%xmm3 \n\t"
                         "mulpd %%xmm2, %%xmm4 \n\t"
                         "mulpd %%xmm1, %%xmm5 \n\t"
                         "mulpd %%xmm2, %%xmm6 \n\t"
                         "mulpd %%xmm1, %%xmm7 \n\t"
                         "mulpd %%xmm2, %%xmm8 \n\t"
                         "mulpd %%xmm1, %%xmm9 \n\t"
                         "mulpd %%xmm2, %%xmm10"
                         :
                         :
                         :
                         "xmm3", "xmm4", "xmm5", "xmm6",
                         "xmm7", "xmm8", "xmm9", "xmm10",
                         "xmm11", "xmm12", "xmm13", "xmm14");

   __asm__ __volatile__ ("addpd %%xmm12, %%xmm11 \n\t"
                         "addpd %%xmm14, %%xmm13 \n\t"
                         "addpd %%xmm4, %%xmm3 \n\t"
                         "addpd %%xmm6, %%xmm5 \n\t"
                         "addpd %%xmm8, %%xmm7 \n\t"
                         "addpd %%xmm10, %%xmm9 \n\t"
                         "movapd %%xmm11, %0 \n\t"
                         "movapd %%xmm13, %1 \n\t"
                         "movapd %%xmm3, %2 \n\t"
                         "movapd %%xmm5, %3 \n\t"
                         "movapd %%xmm7, %4 \n\t"
                         "movapd %%xmm9, %5"
                         :
                         "=m" (w[12]),
                         "=m" (w[13]),
                         "=m" (w[14]),
                         "=m" (w[15]),
                         "=m" (w[16]),
                         "=m" (w[17])
                         :
                         :
                         "xmm3", "xmm5", "xmm7", "xmm9",
                         "xmm11", "xmm13");
}

#else

double tr1_pauli_mat(pauli_dble *A,pauli_dble *B)
{
   double *u,*v,*um;
   double tr0,tr1;

   u=A[0].u;
   v=B[0].u;
   um=u+6;
   tr0=0.0;

   for (;u<um;u++)
   {
      tr0+=u[0]*v[0];
      v+=1;
   }

   um=u+30;
   tr1=0.0;

   for (;u<um;u++)
   {
      tr1+=u[0]*v[0];
      v+=1;
   }

   return tr0+tr1+tr1;
}


void lc3_pauli_mat(double *c,pauli_dble *A1,pauli_dble *A2,pauli_dble *B)
{
   double *u,*v,*w,*um;

   u=A1[0].u;
   v=A2[0].u;
   w=B[0].u;
   um=u+36;

   for (;u<um;u++)
   {
      w[0]=c[1]*u[0]+c[2]*v[0];
      v+=1;
      w+=1;
   }

   w=B[0].u;
   w[0]+=c[0];
   w[1]+=c[0];
   w[2]+=c[0];
   w[3]+=c[0];
   w[4]+=c[0];
   w[5]+=c[0];
}

#endif
