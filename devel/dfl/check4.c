
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2007, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Checks on the deflation projectors
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sw_term.h"
#include "uflds.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "little.h"
#include "dfl.h"
#include "global.h"

int my_rank,id,first,last,step;
int bs[4],Ns,nkv,nmx,eoflg;
double kappa,csw,cF,mu;
double m0,res;
char cnfg_dir[NAME_SIZE],cnfg_file[NAME_SIZE],nbase[NAME_SIZE];


static void new_subspace(void)
{
   int nb,isw,ifail;
   int n,nmax,k,l;
   spinor **mds,**ws;
   sap_parms_t sp;   

   blk_list(SAP_BLOCKS,&nb,&isw);
   
   if (nb==0)
      alloc_bgr(SAP_BLOCKS);

   assign_ud2ubgr(SAP_BLOCKS);
   sw_term(NO_PTS);
   ifail=assign_swd2swbgr(SAP_BLOCKS,ODD_PTS);

   error(ifail!=0,1,"new_subspace [check3.c]",
         "Inversion of the SW term was not safe");

   sp=sap_parms();
   nmax=5;   
   mds=reserve_ws(Ns);
   ws=reserve_ws(1);
   
   for (k=0;k<Ns;k++)
   {
      random_s(VOLUME,mds[k],1.0f);
      bnd_s2zero(ALL_PTS,mds[k]);
   }

   for (n=0;n<nmax;n++)
   {
      for (k=0;k<Ns;k++)
      {
         assign_s2s(VOLUME,mds[k],ws[0]);
         set_s2zero(VOLUME,mds[k]);

         for (l=0;l<sp.ncy;l++)
            sap(0.01f,1,sp.nmr,mds[k],ws[0]);
      }

      for (k=0;k<Ns;k++)
      {
         for (l=0;l<k;l++)
            project(VOLUME,1,mds[k],mds[l]);

         normalize(VOLUME,1,mds[k]);
      }
   }

   dfl_subspace(mds);   
   
   release_ws();
   release_ws();
}


static void random_mode(spinor_dble *sd)
{
   int nb,isw,nv;
   complex_dble **wvd;

   blk_list(DFL_BLOCKS,&nb,&isw);
   nv=nb*Ns;

   wvd=reserve_wvd(1);
   random_vd(nv,wvd[0],1.0);
   vnormalize_dble(nv,1,wvd[0]);
   dfl_vd2sd(wvd[0],sd);
   release_wvd();
}


int main(int argc,char *argv[])
{
   int nsize,icnfg,status;
   double dev0,dev1,dev2,dev3;
   complex z;
   complex_dble zd,w0,w1;
   spinor **ws;
   spinor_dble **wsd;
   lat_parms_t lat;
   sw_parms_t sw;
   tm_parms_t tm;
   dfl_parms_t dfl;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Check on the deflation projectors\n");
      printf("---------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      find_section("Configurations");
      read_line("name","%s",nbase);
      read_line("cnfg_dir","%s",cnfg_dir);
      read_line("first","%d",&first);
      read_line("last","%d",&last);  
      read_line("step","%d",&step);  

      find_section("Lattice parameters");
      read_line("kappa","%lf",&kappa);
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);
      read_line("mu","%lf",&mu);
      read_line("eoflg","%d",&eoflg);
      
      find_section("DFL");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);

      find_section("GCR");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);      
      
      fclose(fin);
   }
   
   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   start_ranlux(0,1234);
   geometry();

   lat=set_lat_parms(6.0,1.0,kappa,0.0,0.0,csw,1.0,cF);
   set_sap_parms(bs,1,4,5);
   m0=lat.m0u;
   sw=set_sw_parms(m0);
   tm=set_tm_parms(eoflg);
   dfl=set_dfl_parms(bs,Ns);

   alloc_ws(Ns+5);
   alloc_wsd(5);
   alloc_wv(2*nkv+2);
   alloc_wvd(4);

   ws=reserve_ws(4);
   wsd=reserve_wsd(4);

   z.re=-1.0f;
   z.im=0.0f;
   zd.re=-1.0;
   zd.im=0.0;
   
   if (my_rank==0)
   {
      printf("kappa = %.6f\n",lat.kappa_u);
      printf("csw = %.6f\n",sw.csw);
      printf("cF = %.6f\n",sw.cF);
      printf("mu = %.6f\n",mu);
      printf("eoflg = %d\n\n",tm.eoflg);
      
      printf("bs = (%d,%d,%d,%d)\n",dfl.bs[0],dfl.bs[1],dfl.bs[2],dfl.bs[3]);
      printf("Ns = %d\n\n",dfl.Ns);

      printf("nkv = %d\n",nkv);
      printf("nmx = %d\n",nmx);      
      printf("res = %.2e\n\n",res);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   error_root(((last-first)%step)!=0,1,"main [check3.c]",
              "last-first is not a multiple of step");
   check_dir_root(cnfg_dir);   

   nsize=name_size("%s/%sn%d",cnfg_dir,nbase,last);
   error_root(nsize>=NAME_SIZE,1,"main [check3.c]",
              "cnfg_dir name is too long");

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file);

      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      } 

      new_subspace();
      
      random_mode(wsd[0]);
      dfl_Ppro_dble(wsd[0],wsd[1]);
      mulc_spinor_add_dble(VOLUME,wsd[0],wsd[1],zd);
      dev0=norm_square_dble(VOLUME,1,wsd[0])/
         norm_square_dble(VOLUME,1,wsd[1]);
      dev0=sqrt(dev0);

      random_sd(VOLUME,wsd[0],1.0);
      dfl_Ppro_dble(wsd[0],wsd[1]);
      dfl_Ppro_dble(wsd[1],wsd[2]);
      mulc_spinor_add_dble(VOLUME,wsd[1],wsd[2],zd);
      dev1=norm_square_dble(VOLUME,1,wsd[1])/
         norm_square_dble(VOLUME,1,wsd[2]);
      dev1=sqrt(dev1);
   
      random_sd(VOLUME,wsd[0],1.0);
      random_sd(VOLUME,wsd[1],1.0);
      assign_sd2sd(VOLUME,wsd[0],wsd[2]);
      assign_sd2sd(VOLUME,wsd[1],wsd[3]);
      dfl_Ppro_dble(wsd[0],wsd[0]);
      dfl_Ppro_dble(wsd[1],wsd[1]);
      w0=spinor_prod_dble(VOLUME,1,wsd[3],wsd[0]);
      w1=spinor_prod_dble(VOLUME,1,wsd[1],wsd[2]);
      w0.re-=w1.re;
      w0.im-=w1.im;
      dev2=(w0.re*w0.re+w0.im*w0.im)/
         (norm_square_dble(VOLUME,1,wsd[3])*norm_square_dble(VOLUME,1,wsd[0]));
      dev2=sqrt(dev2);

      message("Check of dfl_Ppro_dble():\n");
      message("Action in subspace: %.1e\n",dev0);
      message("Projector property: %.1e\n",dev1);
      message("Hermiticity check : %.1e\n\n",dev2);

      random_sd(VOLUME,wsd[0],1.0);
      assign_sd2sd(VOLUME,wsd[0],wsd[3]);
      dfl_Ppro_dble(wsd[0],wsd[1]);
      dfl_Qpro_dble(wsd[0],wsd[2]);
      mulc_spinor_add_dble(VOLUME,wsd[3],wsd[1],zd);
      mulc_spinor_add_dble(VOLUME,wsd[3],wsd[2],zd);
      dev0=norm_square_dble(VOLUME,1,wsd[3])/
         norm_square_dble(VOLUME,1,wsd[0]);
      dev0=sqrt(dev0);

      random_sd(VOLUME,wsd[0],1.0);
      dfl_Ppro_dble(wsd[0],wsd[1]);
      dfl_Qpro_dble(wsd[1],wsd[2]);
      dev1=norm_square_dble(VOLUME,1,wsd[2])/
         norm_square_dble(VOLUME,1,wsd[0]);
      dev1=sqrt(dev1);
   
      message("Check of dfl_Qpro_dble():\n");
      message("Q+P-1: %.1e\n",dev0);
      message("Q*P  : %.1e\n\n",dev1);

      random_sd(VOLUME,wsd[0],1.0);
      random_sd(VOLUME,wsd[1],1.0);
      assign_sd2sd(VOLUME,wsd[0],wsd[2]);
      dfl_Lpro_dble(nkv,nmx,res,mu,wsd[0],wsd[1],&status);
      dfl_Ppro_dble(wsd[0],wsd[3]);
      dev0=norm_square_dble(VOLUME,1,wsd[3])/
         norm_square_dble(VOLUME,1,wsd[0]);
      dev0=sqrt(dev0);

      mulc_spinor_add_dble(VOLUME,wsd[2],wsd[0],zd);
      Dw_dble(mu,wsd[1],wsd[3]);
      mulc_spinor_add_dble(VOLUME,wsd[3],wsd[2],zd);
      dev1=norm_square_dble(VOLUME,1,wsd[3])/
         norm_square_dble(VOLUME,1,wsd[2]);
      dev1=sqrt(dev1);

      random_sd(VOLUME,wsd[0],1.0);
      dfl_Qpro_dble(wsd[0],wsd[0]);
      assign_sd2sd(VOLUME,wsd[0],wsd[2]);   
      dfl_Lpro_dble(nkv,nmx,res,mu,wsd[0],wsd[1],&status);
      mulc_spinor_add_dble(VOLUME,wsd[0],wsd[2],zd);
      dev2=norm_square_dble(VOLUME,1,wsd[0])/
         norm_square_dble(VOLUME,1,wsd[2]);
      dev2=sqrt(dev2);   

      random_mode(wsd[0]);
      Dw_dble(mu,wsd[0],wsd[1]);
      assign_sd2sd(VOLUME,wsd[1],wsd[3]);
      dfl_Lpro_dble(nkv,nmx,res,mu,wsd[1],wsd[2],&status);
      dev3=norm_square_dble(VOLUME,1,wsd[1])/
         norm_square_dble(VOLUME,1,wsd[3]);
      dev3=sqrt(dev3);     
   
      message("Check of dfl_Lpro_dble():\n");
      message("Status = %d\n",status);
      message("P*PL   : %.1e\n",dev0);   
      message("D*phi  : %.1e\n",dev1);   
      message("PL*Q   : %.1e\n",dev2);
      message("PL*D*P : %.1e\n\n",dev3);         

      random_sd(VOLUME,wsd[0],1.0);
      Dw_dble(mu,wsd[0],wsd[1]);
      assign_sd2sd(VOLUME,wsd[1],wsd[2]);      
      dfl_Lpro_dble(nkv,nmx,res,mu,wsd[2],wsd[3],&status);      
      dfl_RLpro_dble(nkv,nmx,res,mu,wsd[0],wsd[1],&status);
      mulc_spinor_add_dble(VOLUME,wsd[2],wsd[1],zd);
      dev0=norm_square_dble(VOLUME,1,wsd[2])/
         norm_square_dble(VOLUME,1,wsd[1]);
      dev0=sqrt(dev0);
   
      Dw_dble(mu,wsd[0],wsd[2]);
      mulc_spinor_add_dble(VOLUME,wsd[2],wsd[1],zd);
      dev1=norm_square_dble(VOLUME,1,wsd[2])/
         norm_square_dble(VOLUME,1,wsd[1]);
      dev1=sqrt(dev1);
   
      message("Check of dfl_RLpro_dble():\n");
      message("Status = %d\n",status);
      message("PL       : %.1e\n",dev0);
      message("D*PR-PL*D: %.1e\n\n",dev1);      

      random_s(VOLUME,ws[0],1.0f);
      assign_s2sd(VOLUME,ws[0],wsd[0]);
      assign_ud2u();
      assign_swd2sw();
      Dw((float)(mu),ws[0],ws[1]);
      Dw_dble(mu,wsd[0],wsd[1]);
      dfl_RLpro_dble(nkv,nmx,res,mu,wsd[0],wsd[1],&status);
      dfl_RLpro(nkv,nmx,FLT_EPSILON,mu,ws[0],ws[1],&status);   
      assign_sd2s(VOLUME,wsd[0],ws[2]);
      assign_sd2s(VOLUME,wsd[1],ws[3]);    

      mulc_spinor_add(VOLUME,ws[2],ws[0],z);
      dev0=(double)(norm_square(VOLUME,1,ws[2])/
                    norm_square(VOLUME,1,ws[0]));
      dev0=sqrt(dev0);   
   
      mulc_spinor_add(VOLUME,ws[3],ws[1],z);
      dev1=(double)(norm_square(VOLUME,1,ws[3])/
                    norm_square(VOLUME,1,ws[1]));
      dev1=sqrt(dev1);

      message("Check of dfl_RLpro():\n");
      message("Status = %d\n",status);
      message("PR: %.1e\n",dev0);
      message("PL: %.1e\n\n",dev1); 
   }
   
   error_chk();   
   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();   
   exit(0);
}
