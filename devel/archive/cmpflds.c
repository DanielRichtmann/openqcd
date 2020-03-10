
/*******************************************************************************
*
* File cmpflds.c
*
* Copyright (C) 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Compare fields of various types.
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "cmpflds.h"
#include "global.h"


static int cmp_su3(su3_dble *u,su3_dble *v)
{
   int it;

   it =((*u).c11.re!=(*v).c11.re);
   it|=((*u).c11.im!=(*v).c11.im);
   it|=((*u).c12.re!=(*v).c12.re);
   it|=((*u).c12.im!=(*v).c12.im);
   it|=((*u).c13.re!=(*v).c13.re);
   it|=((*u).c13.im!=(*v).c13.im);

   it|=((*u).c21.re!=(*v).c21.re);
   it|=((*u).c21.im!=(*v).c21.im);
   it|=((*u).c22.re!=(*v).c22.re);
   it|=((*u).c22.im!=(*v).c22.im);
   it|=((*u).c23.re!=(*v).c23.re);
   it|=((*u).c23.im!=(*v).c23.im);

   it|=((*u).c31.re!=(*v).c31.re);
   it|=((*u).c31.im!=(*v).c31.im);
   it|=((*u).c32.re!=(*v).c32.re);
   it|=((*u).c32.im!=(*v).c32.im);
   it|=((*u).c33.re!=(*v).c33.re);
   it|=((*u).c33.im!=(*v).c33.im);

   return it;
}


static int cmp_su3_alg(su3_alg_dble *X,su3_alg_dble *Y)
{
   int it;

   it =((*X).c1!=(*Y).c1);
   it|=((*X).c2!=(*Y).c2);
   it|=((*X).c3!=(*Y).c3);
   it|=((*X).c4!=(*Y).c4);
   it|=((*X).c5!=(*Y).c5);
   it|=((*X).c6!=(*Y).c6);
   it|=((*X).c7!=(*Y).c7);
   it|=((*X).c8!=(*Y).c8);

   return it;
}


static int cmp_spinor(spinor_dble *X,spinor_dble *Y)
{
   int it;

   it =((*X).c1.c1.re!=(*Y).c1.c1.re);
   it|=((*X).c1.c1.im!=(*Y).c1.c1.im);
   it|=((*X).c1.c2.re!=(*Y).c1.c2.re);
   it|=((*X).c1.c2.im!=(*Y).c1.c2.im);
   it|=((*X).c1.c3.re!=(*Y).c1.c3.re);
   it|=((*X).c1.c3.im!=(*Y).c1.c3.im);

   it|=((*X).c2.c1.re!=(*Y).c2.c1.re);
   it|=((*X).c2.c1.im!=(*Y).c2.c1.im);
   it|=((*X).c2.c2.re!=(*Y).c2.c2.re);
   it|=((*X).c2.c2.im!=(*Y).c2.c2.im);
   it|=((*X).c2.c3.re!=(*Y).c2.c3.re);
   it|=((*X).c2.c3.im!=(*Y).c2.c3.im);

   it|=((*X).c3.c1.re!=(*Y).c3.c1.re);
   it|=((*X).c3.c1.im!=(*Y).c3.c1.im);
   it|=((*X).c3.c2.re!=(*Y).c3.c2.re);
   it|=((*X).c3.c2.im!=(*Y).c3.c2.im);
   it|=((*X).c3.c3.re!=(*Y).c3.c3.re);
   it|=((*X).c3.c3.im!=(*Y).c3.c3.im);

   it|=((*X).c4.c1.re!=(*Y).c4.c1.re);
   it|=((*X).c4.c1.im!=(*Y).c4.c1.im);
   it|=((*X).c4.c2.re!=(*Y).c4.c2.re);
   it|=((*X).c4.c2.im!=(*Y).c4.c2.im);
   it|=((*X).c4.c3.re!=(*Y).c4.c3.re);
   it|=((*X).c4.c3.im!=(*Y).c4.c3.im);

   return it;
}


int check_ud(int vol,su3_dble *ud,su3_dble *vd)
{
   int it;
   su3_dble *um;

   it=0;
   um=ud+vol;

   for (;ud<um;ud++)
   {
      it|=cmp_su3(ud,vd);
      vd+=1;
   }

   return it;
}


int check_fd(int vol,su3_alg_dble *fd,su3_alg_dble *gd)
{
   int it;
   su3_alg_dble *fm;

   it=0;
   fm=fd+vol;

   for (;fd<fm;fd++)
   {
      it|=cmp_su3_alg(fd,gd);
      gd+=1;
   }

   return it;
}


int check_sd(int vol,spinor_dble *sd,spinor_dble *rd)
{
   int it;
   spinor_dble *sm;

   it=0;
   sm=sd+vol;

   for (;sd<sm;sd++)
   {
      it|=cmp_spinor(sd,rd);
      rd+=1;
   }

   return it;
}
