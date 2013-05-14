
/*******************************************************************************
*
* File check7.c
*
* Copyright (C) 2012, 2013 Stefan Schaefer, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Comparison of rwtm*eo() with action4()
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


static double random_pf(void)
{
   mdflds_t *mdfs;

   mdfs=mdflds();
   random_sd(VOLUME/2,(*mdfs).pf[0],1.0);

   set_sd2zero(VOLUME/2,(*mdfs).pf[0]+VOLUME/2);
   bnd_sd2zero(ALL_PTS,(*mdfs).pf[0]);

   return norm_square_dble(VOLUME/2,1,(*mdfs).pf[0]);
}


static void divide_pf(double mu,int isp,int *status)
{
   mdflds_t *mdfs;
   spinor_dble *phi,*chi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;
   
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);
   
   mdfs=mdflds();
   phi=(*mdfs).pf[0];
   sp=solver_parms(isp);
   mu=sqrt(2.0)*mu;
   
   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.res,mu,phi,phi,status);

      error_root(status[0]<0,1,"divide_pf [check7.c]",
                 "CGNE solver failed (parameter set no %d, status = %d)",
                 isp,status[0]);
      
      wsd=reserve_wsd(1);
      chi=wsd[0];
      assign_sd2sd(VOLUME/2,phi,chi);
      Dwhat_dble(-mu,chi,phi);
      mulg5_dble(VOLUME/2,phi);
      set_sd2zero(VOLUME/2,phi+VOLUME/2);      
      release_wsd();
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME/2,phi);
      set_sd2zero(VOLUME/2,phi+VOLUME/2);
      sap_gcr(sp.nkv,sp.nmx,sp.res,mu,phi,phi,status);
      set_sd2zero(VOLUME/2,phi+VOLUME/2);

      error_root(status[0]<0,1,"divide_pf [check7.c]",
                 "SAP_GCR solver failed (parameter set no %d, status = %d)",
                 isp,status[0]);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME/2,phi);
      set_sd2zero(VOLUME/2,phi+VOLUME/2);      
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,mu,phi,phi,status);
      set_sd2zero(VOLUME/2,phi+VOLUME/2);

      error_root((status[0]<0)||(status[1]<0),1,
                 "divide_pf [check7.c]","DFL_SAP_GCR solver failed "
                 "(parameter set no %d, status = (%d,%d,%d))",
                 isp,status[0],status[1],status[2]);
   }
}


int main(int argc,char *argv[])
{
   int my_rank,irw,isp,isp2[3],status[6];
   int isap,idfl,mnkv;
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv;
   double kappa,mu,res;
   double act0,act1,sqn0,sqn1;
   double da,ds,damx,dsmx;
   solver_parms_t sp;
   FILE *flog=NULL,*fin=NULL;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check7.log","w",stdout);
      fin=freopen("check7.in","r",stdin);
      
      printf("\n");
      printf("Comparison of rwtm*() with action4()\n");
      printf("------------------------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      find_section("Solver combination");
      read_line("isp", "%d %d %d",isp2,isp2+1,isp2+2);
   }

   MPI_Bcast(isp2,3,MPI_INT,0,MPI_COMM_WORLD);
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

   set_lat_parms(6.0,1.0,0.0,0.0,0.0,1.234,1.0,1.34);
   set_hmc_parms(0,NULL,1,0,NULL,1,1.0);   

   print_solver_parms(&isap,&idfl);
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

   for (irw=1;irw<3;irw++)
   {
      for (isp=0;isp<3;isp++)
      {
         if (isp==0)
         {
            set_sw_parms(1.0877);         
            mu=1.0;
         }
         else if (isp==1)
         {
            set_sw_parms(0.0877);
            mu=0.1;
         }
         else
         {
            set_sw_parms(-0.0123);         
            mu=0.01;
         }
      
         random_ud();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check7.c]",
                       "dfl_modes failed");
         }      

         start_ranlux(0,8910+isp);
         sqn0=random_pf();

         if (irw==1)
            act0=mu*mu*action4(0.0,0,0,isp2[isp],1,status);
         else
         {
            if ((isp==0)||(isp==1))
               divide_pf(mu,isp2[isp],status+1);
            else
               divide_pf(mu,isp2[isp],status+2);

            act0=mu*mu*mu*mu*action4(0.0,0,0,isp2[isp],1,status);
         }
         
         if (my_rank==0)
         {
            printf("Solver numbers: RWF %d, action %d, mu = %.2e\n",
                   isp,isp2[isp],mu);
            printf("action4(): ");
         
            if ((isp2[isp]==0)||(isp2[isp]==1))
               printf("status = %d\n",status[0]);
            else if (isp2[isp]==2)
               printf("status = (%d,%d,%d)\n",
                      status[0],status[1],status[2]);
         }
      
         start_ranlux(0,8910+isp);

         if (irw==1)
            act1=rwtm1eo(mu,isp,&sqn1,status);
         else
            act1=rwtm2eo(mu,isp,&sqn1,status);
         
         da=fabs(1.0-act1/act0);
         ds=fabs(1.0-sqn1/sqn0);

         if (da>damx)
            damx=da;
         if (ds>dsmx)
            dsmx=ds;

         if (my_rank==0)
         {
            if (irw==1)
            {
               printf("rwtm1eo(): ");

               if ((isp==0)||(isp==1))
                  printf("status = %d\n",status[0]);
               else if (isp==2)
                  printf("status = (%d,%d,%d)\n",
                         status[0],status[1],status[2]);
            }
            else
            {
               printf("rwtm2eo(): ");

               if ((isp==0)||(isp==1))
                  printf("status = %d,%d\n",status[0],status[1]);
               else if (isp==2)
                  printf("status = (%d,%d,%d),(%d,%d,%d)\n",
                         status[0],status[1],status[2],status[3],
                         status[4],status[5]);
            }
            
            printf("|1-act1/act0| = %.1e, |1-sqn1/sqn0| = %.1e\n\n",da,ds);
         }      
      
         error_chk();
      }
   }
   
   if (my_rank==0)
   {
      printf("max|1-act1/act0| = %.1e, max|1-sqn1/sqn0| = %.1e\n\n",
             damx,dsmx);
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
