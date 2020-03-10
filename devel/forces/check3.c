
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005-2016, 2018 Martin Luescher, Filippo Palombi
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the programs force0() and action0().
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
#include "linalg.h"
#include "forces.h"
#include "auxfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)


static qflt dSdt(double c)
{
   mdflds_t *mdfs;

   mdfs=mdflds();
   check_bnd_fld((*mdfs).mom);

   force0(c);
   check_bnd_fld((*mdfs).frc);

   return scalar_prod_alg(4*VOLUME,1,(*mdfs).mom,(*mdfs).frc);
}


static double chk_chs(double c)
{
   double dev;
   qflt rqsm;
   su3_alg_dble **wfd;
   mdflds_t *mdfs;

   wfd=reserve_wfd(1);
   mdfs=mdflds();

   random_ud();
   force0(c);
   assign_alg2alg(4*VOLUME,(*mdfs).frc,wfd[0]);

   set_ud_phase();
   force0(c);
   muladd_assign_alg(4*VOLUME,-1.0,(*mdfs).frc,wfd[0]);
   rqsm=norm_square_alg(4*VOLUME,1,wfd[0]);
   dev=rqsm.q[0];
   rqsm=norm_square_alg(4*VOLUME,1,(*mdfs).frc);
   dev/=rqsm.q[0];
   release_wfd();

   return sqrt(dev);
}


int main(int argc,char *argv[])
{
   int my_rank,k,iact,ie,bc;
   double c,eps;
   double dev_frc,sig_loss;
   double phi[2],phi_prime[2],theta[3];
   qflt dsdt,act,act0,act1;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);

      printf("\n");
      printf("Check of the programs force0() and action0()\n");
      printf("--------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(3.5,0.33,0,NULL,0,1.0);
   print_lat_parms();

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;

   iact=0;
   set_hmc_parms(1,&iact,0,0,NULL,1,1.0);
   set_bc_parms(bc,0.9012,1.2034,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(3);

   start_ranlux(0,1234);
   geometry();
   alloc_wfd(1);
   c=0.789;
   dev_frc=chk_chs(c);

   if (my_rank==0)
      printf("Deviation of gauge force after a phase change = %.1e\n\n",
             dev_frc);

   for (k=0;k<4;k++)
   {
      random_ud();
      set_ud_phase();
      random_mom();
      dsdt=dSdt(c);

      eps=2.0e-4;
      rot_ud(eps);
      act0=action0(1);
      scl_qflt(2.0/3.0,act0.q);
      rot_ud(-eps);

      rot_ud(-eps);
      act1=action0(1);
      scl_qflt(-2.0/3.0,act1.q);
      rot_ud(eps);

      rot_ud(2.0*eps);
      act=action0(1);
      scl_qflt(-1.0/12.0,act.q);
      add_qflt(act0.q,act.q,act0.q);
      rot_ud(-2.0*eps);

      rot_ud(-2.0*eps);
      act=action0(1);
      scl_qflt(1.0/12.0,act.q);
      add_qflt(act1.q,act.q,act1.q);
      rot_ud(2.0*eps);

      scl_qflt(c,act0.q);
      scl_qflt(c,act1.q);

      add_qflt(act0.q,act1.q,act.q);
      sig_loss=-log10(fabs(act.q[0]/act0.q[0]));

      scl_qflt(-1.0/eps,act.q);
      add_qflt(dsdt.q,act.q,act.q);
      dev_frc=act.q[0]/dsdt.q[0];

      unset_ud_phase();
      ie=check_bc(0.0);
      error_root(ie!=1,1,"main [check3.c]",
                 "Operations did not preserve boundary conditions");

      if (my_rank==0)
      {
         printf("Relative deviation of dS/dt = %.2e ",fabs(dev_frc));
         printf("[significance loss = %d digits]\n",(int)(sig_loss));
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
