
/*******************************************************************************
*
* File check6.c
*
* Copyright (C) 2011-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of force1() and action1().
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
#include "dfl.h"
#include "forces.h"
#include "auxfcts.h"
#include "global.h"

#define N0 (NPROC0*L0)


static double check_rotpf(double mu,int ipf)
{
   double nrm;
   spinor_dble *phi;
   mdflds_t *mdfs;

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];

   save_ranlux();
   (void)(setpf1(mu,ipf,0));
   nrm=unorm_dble(VOLUME,1,phi);

   restore_ranlux();
   rotpf1(mu,ipf,0,0,1.234,-1.234,NULL);

   return unorm_dble(VOLUME,1,phi)/nrm;
}


static qflt dSdt(double mu,int ipf,int isp,int *status)
{
   mdflds_t *mdfs;

   mdfs=mdflds();
   set_frc2zero();
   force1(mu,ipf,isp,0,1.2345,status);
   check_bnd_fld((*mdfs).frc);

   return scalar_prod_alg(4*VOLUME,1,(*mdfs).mom,(*mdfs).frc);
}


int main(int argc,char *argv[])
{
   int my_rank,bc,is,isp,mnkv,status[6];
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv;
   int i,isap,idfl,idmy;
   double chi[2],chi_prime[2],theta[3];
   double kappa,mu,res;
   double dev,eps,*qact[1];
   double dev_act[2],dev_frc,sig_loss,rdmy;
   qflt dsdt,act0,act1,act;
   solver_parms_t sp;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check6.log","w",stdout);
      fin=freopen("check6.in","r",stdin);

      printf("\n");
      printf("Check of force1() and action1()\n");
      printf("-------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check6.c]",
                    "Syntax: check6 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check6.c]",
                    "Syntax: check6 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,is,1.782);
   print_lat_parms();

   chi[0]=0.123;
   chi[1]=-0.534;
   chi_prime[0]=0.912;
   chi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc,1.0,1.0,0.953,1.203,chi,chi_prime,theta);
   print_bc_parms(2);

   if (my_rank==0)
   {
      find_section("SAP");
      read_iprms("bs",4,bs);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_iprms("bs",4,bs);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_parms(bs,Ns);

   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy);

   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);

   idmy=0;
   rdmy=0.0;
   set_action_parms(0,ACF_TM1,0,0,NULL,&idmy,&idmy);
   set_hmc_parms(1,&idmy,1,1,&rdmy,1,1.0);
   mnkv=0;

   for (isp=0;isp<3;isp++)
   {
      read_solver_parms(isp);
      sp=solver_parms(isp);

      if (sp.nkv>mnkv)
         mnkv=sp.nkv;
   }

   if (my_rank==0)
      fclose(fin);

   print_solver_parms(&isap,&idfl);
   print_sap_parms(1);
   print_dfl_parms(0);

   start_ranlux(0,1245);
   geometry();

   set_sw_parms(-0.0123);
   mnkv=2*mnkv+2;
   if (mnkv<(Ns+2))
      mnkv=Ns+2;
   if (mnkv<5)
      mnkv=5;

   alloc_ws(mnkv);
   alloc_wsd(6);
   alloc_wv(2*nkv+2);
   alloc_wvd(4);

   for (isp=0;isp<3;isp++)
   {
      if (isp==0)
      {
         mu=1.0;
         eps=1.0e-4;
      }
      else if (isp==1)
      {
         mu=0.1;
         eps=2.0e-4;
      }
      else
      {
         mu=0.01;
         eps=3.0e-4;
      }

      random_ud();
      set_ud_phase();
      random_mom();

      if (isp==2)
      {
         dfl_modes(status);
         error_root(status[0]<0,1,"main [check6.c]",
                    "dfl_modes failed");
      }

      for (i=0;i<6;i++)
         status[i]=0;

      dev=check_rotpf(mu,0);

      act0=setpf1(mu,0,0);
      act1=action1(mu,0,isp,0,0,status);
      error_root((status[0]<0)||(status[1]<0),1,"main [check6.c]",
                 "action1 failed (mu = %.2e, isp=%d)",mu,isp);
      act.q[0]=-act1.q[0];
      act.q[1]=-act1.q[1];
      add_qflt(act0.q,act.q,act.q);
      rdmy=fabs(act.q[0]);
      MPI_Reduce(&rdmy,dev_act,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      qact[0]=act.q;
      global_qsum(1,qact,qact);
      dev_act[1]=act.q[0];

      dsdt=dSdt(mu,0,isp,status);

      if (my_rank==0)
      {
         printf("Solver number %d, mu = %.2e\n",isp,mu);

         if (isp==0)
            printf("Status = %d\n",status[0]);
         else if (isp==1)
            printf("Status = %d,%d\n",status[0],status[1]);
         else
            printf("Status = (%d,%d,%d),(%d,%d,%d)\n",
                   status[0],status[1],status[2],status[3],
                   status[4],status[5]);

         printf("Absolute action difference |setpf1-action1| = %.1e,",
                fabs(dev_act[1]));
         printf(" %.1e (local)\n",dev_act[0]);
         printf("Check of rotpf1 = %.1e\n",dev);
         fflush(flog);
      }

      rot_ud(eps);
      act0=action1(mu,0,isp,0,1,status);
      scl_qflt(2.0/3.0,act0.q);
      rot_ud(-eps);

      rot_ud(-eps);
      act1=action1(mu,0,isp,0,1,status);
      scl_qflt(-2.0/3.0,act1.q);
      rot_ud(eps);

      rot_ud(2.0*eps);
      act=action1(mu,0,isp,0,1,status);
      scl_qflt(-1.0/12.0,act.q);
      add_qflt(act0.q,act.q,act0.q);
      rot_ud(-2.0*eps);

      rot_ud(-2.0*eps);
      act=action1(mu,0,isp,0,1,status);
      scl_qflt(1.0/12.0,act.q);
      add_qflt(act1.q,act.q,act1.q);
      rot_ud(2.0*eps);

      add_qflt(act0.q,act1.q,act.q);
      sig_loss=-log10(fabs(act.q[0]/act0.q[0]));
      scl_qflt(-1.2345/eps,act.q);
      add_qflt(dsdt.q,act.q,act.q);
      dev_frc=fabs(act.q[0]/dsdt.q[0]);

      if (my_rank==0)
      {
         printf("Relative deviation of dS/dt = %.2e ",dev_frc);
         printf("[significance loss = %d digits]\n\n",(int)(sig_loss));
         fflush(flog);
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
