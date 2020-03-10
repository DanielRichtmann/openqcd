
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2005-2016, 2018 Martin Luescher, Filippo Palombi,
*                               Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of sw_frc() and hop_frc().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "forces.h"
#include "auxfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)


static int is_Xt_zero(u3_alg_dble *X)
{
   int ie;

   ie=1;
   ie&=((*X).c1==0.0);
   ie&=((*X).c2==0.0);
   ie&=((*X).c3==0.0);
   ie&=((*X).c4==0.0);
   ie&=((*X).c5==0.0);
   ie&=((*X).c6==0.0);
   ie&=((*X).c7==0.0);
   ie&=((*X).c8==0.0);
   ie&=((*X).c9==0.0);

   return ie;
}


static int is_Xv_zero(su3_dble *X)
{
   int ie;

   ie=1;
   ie&=((*X).c11.re==0.0);
   ie&=((*X).c11.im==0.0);
   ie&=((*X).c12.re==0.0);
   ie&=((*X).c12.im==0.0);
   ie&=((*X).c13.re==0.0);
   ie&=((*X).c13.im==0.0);

   ie&=((*X).c21.re==0.0);
   ie&=((*X).c21.im==0.0);
   ie&=((*X).c22.re==0.0);
   ie&=((*X).c22.im==0.0);
   ie&=((*X).c23.re==0.0);
   ie&=((*X).c23.im==0.0);

   ie&=((*X).c31.re==0.0);
   ie&=((*X).c31.im==0.0);
   ie&=((*X).c32.re==0.0);
   ie&=((*X).c32.im==0.0);
   ie&=((*X).c33.re==0.0);
   ie&=((*X).c33.im==0.0);

   return ie;
}


static void check_Xtbnd(ptset_t set)
{
   int bc,ix,t,n,ie;
   int ia,ib;
   u3_alg_dble **xt;

   bc=bc_type();
   xt=xtensor();
   ie=0;
   ia=0;
   ib=VOLUME;

   if (set==EVEN_PTS)
      ib=(VOLUME/2);
   else if (set==ODD_PTS)
      ia=(VOLUME/2);
   else if (set==NO_PTS)
      ia=VOLUME;

   for (ix=0;ix<VOLUME;ix++)
   {
      if ((ix>=ia)&&(ix<ib))
      {
         t=global_time(ix);

         if (((t==0)&&(bc!=3))||((t==(N0-1))&&(bc==0)))
         {
            for (n=0;n<6;n++)
            {
               ie|=(is_Xt_zero(xt[n])^0x1);
               xt[n]+=1;
            }
         }
         else
         {
            for (n=0;n<6;n++)
            {
               ie|=is_Xt_zero(xt[n]);
               xt[n]+=1;
            }
         }
      }
      else
      {
         for (n=0;n<6;n++)
         {
            ie|=(is_Xt_zero(xt[n])^0x1);
            xt[n]+=1;
         }
      }
   }

   error(ie!=0,1,"check_Xtbnd [check4.c]",
         "X tensor field vanishes on an incorrect set of points");
}


static void check_Xvbnd(void)
{
   int bc,ix,t,ifc,ie;
   su3_dble *xv;

   bc=bc_type();
   xv=xvector();
   ie=0;

   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (((t==0)&&(bc!=3))||((t==(N0-1))&&(bc==0)))
      {
         for (ifc=0;ifc<8;ifc++)
         {
            ie|=(is_Xv_zero(xv)^0x1);
            xv+=1;
         }
      }
      else if ((t==1)&&(bc!=3))
      {
         ie|=is_Xv_zero(xv);
         xv+=1;

         ie|=(is_Xv_zero(xv)^0x1);
         xv+=1;

         for (ifc=2;ifc<8;ifc++)
         {
            ie|=is_Xv_zero(xv);
            xv+=1;
         }
      }
      else if (((t==(N0-2))&&(bc==0))||((t==(N0-1))&&(bc!=3)))
      {
         ie|=(is_Xv_zero(xv)^0x1);
         xv+=1;

         for (ifc=1;ifc<8;ifc++)
         {
            ie|=is_Xv_zero(xv);
            xv+=1;
         }
      }
      else
      {
         for (ifc=0;ifc<8;ifc++)
         {
            ie|=is_Xv_zero(xv);
            xv+=1;
         }
      }
   }

   error(ie!=0,1,"check_Xvbnd [check4.c]",
         "X vector field vanishes on an incorrect set of links");
}


static qflt action(int k,spinor_dble **phi)
{
   int l;
   qflt act;
   spinor_dble **wsd;

   wsd=reserve_wsd(2);
   sw_term(NO_PTS);
   assign_sd2sd(VOLUME,phi[0],wsd[0]);

   for (l=0;l<k;l++)
   {
      Dw_dble(0.0,wsd[0],wsd[1]);
      mulg5_dble(VOLUME,wsd[1]);
      scale_dble(VOLUME,0.125,wsd[1]);
      assign_sd2sd(VOLUME,wsd[1],wsd[0]);
   }

   act=spinor_prod_re_dble(VOLUME,1,phi[0],wsd[0]);
   release_wsd();

   return act;
}


static qflt dSdt(int k,spinor_dble **phi)
{
   int l;
   spinor_dble **wsd;
   mdflds_t *mdfs;

   mdfs=mdflds();
   wsd=reserve_wsd(k);
   sw_term(NO_PTS);
   assign_sd2sd(VOLUME,phi[0],wsd[0]);

   for (l=1;l<k;l++)
   {
      Dw_dble(0.0,wsd[l-1],wsd[l]);
      mulg5_dble(VOLUME,wsd[l]);
      scale_dble(VOLUME,0.125,wsd[l]);
   }

   set_frc2zero();
   set_xt2zero();
   set_xv2zero();

   for (l=0;l<k;l++)
   {
      add_prod2xt(-0.0625,wsd[l],wsd[k-l-1]);
      add_prod2xv(-0.0625,wsd[l],wsd[k-l-1]);
   }

   check_Xtbnd(ALL_PTS);
   check_Xvbnd();

   sw_frc(1.0);
   hop_frc(1.0);
   check_bnd_fld((*mdfs).frc);
   release_wsd();

   return scalar_prod_alg(4*VOLUME,1,(*mdfs).mom,(*mdfs).frc);
}


static qflt action_det(ptset_t set)
{
   int bc,ie,io;
   int vol,ofs,ix,im,t;
   double c,p,*qsm[1];
   qflt rqsm;
   complex_dble z;
   pauli_dble *m;
   sw_parms_t swp;

   rqsm.q[0]=0.0;
   rqsm.q[1]=0.0;

   if (set==NO_PTS)
      return rqsm;

   bc=bc_type();
   swp=sw_parms();

   if ((4.0+swp.m0)>1.0)
      c=pow(4.0+swp.m0,-6.0);
   else
      c=1.0;

   if (query_flags(SWD_UP2DATE)!=1)
      sw_term(NO_PTS);
   else
   {
      ie=query_flags(SWD_E_INVERTED);
      io=query_flags(SWD_O_INVERTED);

      if (((ie==1)&&((set==ALL_PTS)||(set==EVEN_PTS)))||
          ((io==1)&&((set==ALL_PTS)||(set==ODD_PTS))))
         sw_term(NO_PTS);
   }

   if (set==ODD_PTS)
      ofs=(VOLUME/2);
   else
      ofs=0;

   if (set==EVEN_PTS)
      vol=(VOLUME/2);
   else
      vol=VOLUME;

   ix=ofs;
   m=swdfld()+2*ofs;

   while (ix<vol)
   {
      im=ix+8;
      if (im>vol)
         im=vol;
      p=1.0;

      for (;ix<im;ix++)
      {
         t=global_time(ix);

         if (((t>0)||(bc==3))&&((t<(N0-1))||(bc!=0)))
         {
            z=det_pauli_dble(0.0,m);
            p*=(c*z.re);
            z=det_pauli_dble(0.0,m+1);
            p*=(c*z.re);
         }

         m+=2;
      }

      acc_qflt(-log(fabs(p)),rqsm.q);
   }

   rqsm.q[0]*=2.0;
   rqsm.q[1]*=2.0;

   if (NPROC>1)
   {
      qsm[0]=rqsm.q;
      global_qsum(1,qsm,qsm);
   }

   return rqsm;
}


static qflt dSdt_det(ptset_t set)
{
   int ifail;
   mdflds_t *mdfs;

   mdfs=mdflds();
   set_xt2zero();
   ifail=add_det2xt(2.0,set);
   error_root(ifail!=0,1,"dSdt_det [check4.c]",
              "Inversion of the SW term was not safe");
   check_Xtbnd(set);

   set_frc2zero();
   sw_frc(1.0);

   if (set==ALL_PTS)
      check_bnd_fld((*mdfs).frc);

   return scalar_prod_alg(4*VOLUME,1,(*mdfs).mom,(*mdfs).frc);
}


int main(int argc,char *argv[])
{
   int my_rank,iact,bc,k;
   double chi[2],chi_prime[2],theta[3];
   double eps,dev_frc,sig_loss;
   qflt dsdt,act0,act1,act;
   spinor_dble **phi;
   ptset_t set;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);

      printf("\n");
      printf("Check of sw_frc() and hop_frc()\n");
      printf("-------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check4.c]",
                    "Syntax: check4 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,0,1.782);
   print_lat_parms();

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   chi[0]=0.123;
   chi[1]=-0.534;
   chi_prime[0]=0.912;
   chi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;

   iact=0;
   set_hmc_parms(1,&iact,0,0,NULL,1,1.0);
   set_bc_parms(bc,1.0,1.0,0.953,1.203,chi,chi_prime,theta);
   print_bc_parms(1);

   start_ranlux(0,1245);
   geometry();

   set_sw_parms(-0.0123);
   alloc_wsd(6);
   phi=reserve_wsd(1);

   for (k=1;k<=4;k++)
   {
      random_ud();
      set_ud_phase();
      random_mom();
      random_sd(VOLUME,phi[0],1.0);
      bnd_sd2zero(ALL_PTS,phi[0]);
      dsdt=dSdt(k,phi);

      eps=5.0e-5;
      rot_ud(eps);
      act0=action(k,phi);
      scl_qflt(2.0/3.0,act0.q);
      rot_ud(-eps);

      rot_ud(-eps);
      act1=action(k,phi);
      scl_qflt(-2.0/3.0,act1.q);
      rot_ud(eps);

      rot_ud(2.0*eps);
      act=action(k,phi);
      scl_qflt(-1.0/12.0,act.q);
      add_qflt(act0.q,act.q,act0.q);
      rot_ud(-2.0*eps);

      rot_ud(-2.0*eps);
      act=action(k,phi);
      scl_qflt(1.0/12.0,act.q);
      add_qflt(act1.q,act.q,act1.q);
      rot_ud(2.0*eps);

      add_qflt(act0.q,act1.q,act.q);
      sig_loss=-log10(fabs(act.q[0]/act0.q[0]));
      scl_qflt(-1.0/eps,act.q);
      add_qflt(dsdt.q,act.q,act.q);
      dev_frc=fabs(act.q[0]/dsdt.q[0]);

      if (my_rank==0)
      {
         printf("Calculation of the force for S=(phi,Q^%d*phi):\n",k);
         printf("Relative deviation of dS/dt = %.2e ",dev_frc);
         printf("[significance loss = %d digits]\n\n",(int)(sig_loss));
      }
   }

   if (my_rank==0)
      printf("Calculation of the force for S=-2*Tr{ln(SW term)}:\n");

   for (k=0;k<4;k++)
   {
      if (k==0)
         set=NO_PTS;
      else if (k==1)
         set=EVEN_PTS;
      else if (k==2)
         set=ODD_PTS;
      else
         set=ALL_PTS;

      random_ud();
      set_ud_phase();
      random_mom();
      dsdt=dSdt_det(set);

      eps=5.0e-4;
      rot_ud(eps);
      act0=action_det(set);
      scl_qflt(2.0/3.0,act0.q);
      rot_ud(-eps);

      rot_ud(-eps);
      act1=action_det(set);
      scl_qflt(-2.0/3.0,act1.q);
      rot_ud(eps);

      rot_ud(2.0*eps);
      act=action_det(set);
      scl_qflt(-1.0/12.0,act.q);
      add_qflt(act0.q,act.q,act0.q);
      rot_ud(-2.0*eps);

      rot_ud(-2.0*eps);
      act=action_det(set);
      scl_qflt(1.0/12.0,act.q);
      add_qflt(act1.q,act.q,act1.q);
      rot_ud(2.0*eps);

      add_qflt(act0.q,act1.q,act.q);
      if (k>0)
         sig_loss=-log10(fabs(act.q[0]/act0.q[0]));
      scl_qflt(-1.0/eps,act.q);
      add_qflt(dsdt.q,act.q,act.q);

      if (k>0)
         dev_frc=fabs(act.q[0]/dsdt.q[0]);
      else
         dev_frc=fabs(act.q[0]);

      if (my_rank==0)
      {
         if (k==0)
            printf("set=NO_PTS:   ");
         else if (k==1)
            printf("set=EVEN_PTS: ");
         else if (k==2)
            printf("set=ODD_PTS:  ");
         else
            printf("set=ALL_PTS:  ");

         if (k>0)
         {
            printf("relative deviation of dS/dt = %.2e ",dev_frc);
            printf("[significance loss = %d digits]\n",(int)(sig_loss));
         }
         else
            printf("absolute deviation of dS/dt = %.2e\n",dev_frc);
      }
   }

   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
