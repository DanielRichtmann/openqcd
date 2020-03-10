
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2012-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Comparison of rwtm*() with action1().
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
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
#include "global.h"


static qflt random_pf(void)
{
   qflt nrm;
   spinor_dble *phi,**wsd;
   mdflds_t *mdfs;

   wsd=reserve_wsd(1);
   phi=wsd[0];
   random_sd(VOLUME,phi,1.0);
   bnd_sd2zero(ALL_PTS,phi);
   nrm=norm_square_dble(VOLUME,1,phi);

   mdfs=mdflds();
   assign_sd2sd(VOLUME,phi,(*mdfs).pf[0]);
   release_wsd();

   return nrm;
}


static void divide_pf(double mu,int isp,int *status)
{
   spinor_dble *phi,**wsd;
   spinor_dble *chi,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;

   wsd=reserve_wsd(1);
   phi=wsd[0];
   mdfs=mdflds();
   assign_sd2sd(VOLUME,(*mdfs).pf[0],phi);
   sp=solver_parms(isp);

   if (sp.solver==CGNE)
   {
      tmcg(sp.nmx,sp.istop,sp.res,mu,phi,phi,status);

      error_root(status[0]<0,1,"divide_pf [check5.c]",
                 "CGNE solver failed (parameter set no %d, status = %d)",
                 isp,status[0]);

      rsd=reserve_wsd(1);
      chi=rsd[0];
      assign_sd2sd(VOLUME,phi,chi);
      Dw_dble(-mu,chi,phi);
      mulg5_dble(VOLUME,phi);
      release_wsd();
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME,phi);
      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,phi,phi,status);

      error_root(status[0]<0,1,"divide_pf [check5.c]",
                 "SAP_GCR solver failed (parameter set no %d, status = %d)",
                 isp,status[0]);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME,phi);
      dfl_sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,phi,phi,status);

      error_root((status[0]<0)||(status[1]<0),1,
                 "divide_pf [check5.c]","DFL_SAP_GCR solver failed "
                 "(parameter set no %d, status = (%d,%d,%d))",
                 isp,status[0],status[1],status[2]);
   }

   assign_sd2sd(VOLUME,phi,(*mdfs).pf[0]);
   release_wsd();
}


int main(int argc,char *argv[])
{
   int my_rank,bc,irw,isp,status[6],mnkv;
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv,idmy;
   double chi[2],chi_prime[2],theta[3];
   double kappa,mu,mu1,mu2,mu1sq,mu2sq,res;
   double da,ds,damx,dsmx,rdmy;
   qflt act0,act1,sqn0,sqn1;
   solver_parms_t sp;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check5.in","r",stdin);

      printf("\n");
      printf("Comparison of rwtm*() with action1()\n");
      printf("------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check5.c]",
                    "Syntax: check5 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,0,1.782);
   print_lat_parms();

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   chi[0]=0.123;
   chi[1]=-0.534;
   chi_prime[0]=0.912;
   chi_prime[1]=0.078;
   theta[0]=0.34;
   theta[1]=-1.25;
   theta[2]=0.58;
   set_bc_parms(bc,1.0,1.0,0.953,1.203,chi,chi_prime,theta);
   print_bc_parms(2);

   mnkv=0;

   for (isp=0;isp<3;isp++)
   {
      read_solver_parms(isp);
      sp=solver_parms(isp);

      if (sp.nkv>mnkv)
         mnkv=sp.nkv;
   }

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,0,1,1);

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
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
      fclose(fin);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);

   idmy=0;
   rdmy=0.0;
   set_action_parms(0,ACF_TM1,0,0,NULL,&idmy,&idmy);
   set_hmc_parms(1,&idmy,1,1,&rdmy,1,1.0);

   print_solver_parms(status,status+1);
   print_sap_parms(0);
   print_dfl_parms(0);

   start_ranlux(0,1245);
   geometry();

   mnkv=2*mnkv+2;
   if (mnkv<(Ns+2))
      mnkv=Ns+2;
   if (mnkv<5)
      mnkv=5;

   alloc_ws(mnkv);
   alloc_wsd(6);
   alloc_wv(2*nkv+2);
   alloc_wvd(4);
   damx=0.0;
   dsmx=0.0;

   for (irw=1;irw<5;irw++)
   {
      for (isp=0;isp<3;isp++)
      {
         if (isp==0)
         {
            set_sw_parms(1.0877);
            if (irw<3)
               mu1=1.0;
            else
               mu1=0.0;
            mu2=1.23;
         }
         else if (isp==1)
         {
            set_sw_parms(0.0877);
            if (irw<3)
               mu1=0.1;
            else
               mu1=0.0;
            mu2=0.123;
         }
         else
         {
            set_sw_parms(-0.0123);
            if (irw<3)
               mu1=0.01;
            else
               mu1=0.0;
            mu2=0.0123;
         }

         mu1sq=mu1*mu1;
         mu2sq=mu2*mu2;

         random_ud();
         set_ud_phase();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check5.c]",
                       "dfl_modes failed");
         }

         start_ranlux(0,8910+isp);
         sqn0=random_pf();

         if ((irw&0x1)==1)
         {
            act0=action1(mu1,0,isp,0,1,status);
            scl_qflt(mu2sq-mu1sq,act0.q);
         }
         else
         {
            if ((isp==0)||(isp==1))
               divide_pf(mu1,isp,status+1);
            else
               divide_pf(mu1,isp,status+3);

            act0=action1(mu1,0,isp,0,1,status);
            scl_qflt(mu1sq*(mu2sq-mu1sq),act0.q);
            act1=action1(sqrt(2.0)*mu2,0,isp,0,1,status);
            scl_qflt(2.0*mu2sq*mu2sq,act1.q);
            add_qflt(act1.q,act0.q,act0.q);
            scl_qflt((mu2sq-mu1sq)/(2.0*mu2sq-mu1sq),act0.q);
         }

         if (my_rank==0)
         {
            printf("Solver number %d, mu1 = %.2e, mu2 = %.2e\n",isp,mu1,mu2);
            printf("action1(): ");

            if ((isp==0)||(isp==1))
               printf("status = %d\n",status[0]);
            else if (isp==2)
               printf("status = (%d,%d,%d)\n",
                      status[0],status[1],status[2]);
         }

         start_ranlux(0,8910+isp);

         if ((irw&0x1)==1)
            act1=rwtm1(mu1,mu2,isp,&sqn1,status);
         else
            act1=rwtm2(mu1,mu2,isp,&sqn1,status);

         act1.q[0]=-act1.q[0];
         act1.q[1]=-act1.q[1];
         sqn1.q[0]=-sqn1.q[0];
         sqn1.q[1]=-sqn1.q[1];
         add_qflt(act0.q,act1.q,act1.q);
         add_qflt(sqn0.q,sqn1.q,sqn1.q);

         da=fabs(act1.q[0]/act0.q[0]);
         ds=fabs(sqn1.q[0]/sqn0.q[0]);

         if (da>damx)
            damx=da;
         if (ds>dsmx)
            dsmx=ds;

         if (my_rank==0)
         {
            if ((irw&0x1)==1)
            {
               printf("rwtm1(): ");

               if ((isp==0)||(isp==1))
                  printf("status = %d\n",status[0]);
               else if (isp==2)
                  printf("status = (%d,%d,%d)\n",
                         status[0],status[1],status[2]);
            }
            else
            {
               printf("rwtm2(): ");

               if ((isp==0)||(isp==1))
                  printf("status = %d,%d\n",status[0],status[1]);
               else if (isp==2)
                  printf("status = (%d,%d,%d),(%d,%d,%d)\n",
                         status[0],status[1],status[2],status[3],
                         status[4],status[5]);
            }

            printf("|1-act1/act0| = %.1e, |1-sqn1/sqn0| = %.1e\n\n",da,ds);
         }
      }
   }

   if (my_rank==0)
   {
      printf("max|1-act1/act0| = %.1e, max|1-sqn1/sqn0| = %.1e\n\n",damx,dsmx);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
