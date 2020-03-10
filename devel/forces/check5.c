
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2011-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the CG solver.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "forces.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
   int type,nio_nodes,nio_streams;
   int nb,ib;
   char cnfg_dir[NAME_SIZE];
} iodat;

static int my_rank,first,last,step,nmx;
static double mu,res;
static char line[NAME_SIZE];
static char nbase[NAME_SIZE],cnfg_file[NAME_SIZE];
static FILE *flog=NULL,*fin=NULL;


static void read_iodat(void)
{
   int type,nion,nios;

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Configurations");
      read_line("type","%s",line);

      if (strchr(line,'e')!=NULL)
         type=0x1;
      else if (strchr(line,'b')!=NULL)
         type=0x2;
      else if (strchr(line,'l')!=NULL)
         type=0x4;
      else
         type=0x0;

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [check5.c]",
                 "Improper configuration storage type");

      read_line("cnfg_dir","%s",line);

      if (type&0x6)
      {
         read_line("nio_nodes","%d",&nion);
         read_line("nio_streams","%d",&nios);
      }
      else
      {
         nion=1;
         nios=0;
      }

      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_iodat [check5.c]","Improper configuration range");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(line,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat.type=type;
   strcpy(iodat.cnfg_dir,line);
   iodat.nio_nodes=nion;
   iodat.nio_streams=nios;
}


static void read_lat_parms(void)
{
   int isw;
   double kappa,csw;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("kappa","%lf",&kappa);
      read_line("isw","%d",&isw);
      read_line("csw","%lf",&csw);
      read_line("mu","%lf",&mu);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   (void)(set_lat_parms(5.5,1.0,1,&kappa,isw,csw));
}


static void read_bc_parms(void)
{
   int bc;
   double phi[2],phi_prime[2],theta[3];
   double cF,cFp;

   if (my_rank==0)
   {
      find_section("Boundary conditions");
      read_line("type","%d",&bc);

      cF=1.0;
      cFp=1.0;
      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;

      if (bc==1)
         read_dprms("phi",2,phi);

      if ((bc==1)||(bc==2))
         read_dprms("phi'",2,phi_prime);

      if (bc<3)
      {
         read_line("cF","%lf",&cF);

         if (bc==2)
            read_line("cF'","%lf",&cFp);
         else
            cFp=cF;
      }

      read_dprms("theta",3,theta);
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cFp,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(theta,3,MPI_DOUBLE,0,MPI_COMM_WORLD);

   (void)(set_bc_parms(bc,1.0,1.0,cF,cFp,phi,phi_prime,theta));
}


static void read_CG_parms(void)
{
   if (my_rank==0)
   {
      find_section("CG");
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static void check_files(void)
{
   int type,nion,nb,ib,n;
   int ns[4],bs[4];
   char *cnfg_dir;

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;
   iodat.nb=0;
   iodat.ib=NPROC;

   if (type&0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
            "check_files [check5.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);

      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check5.c]","Lattice size mismatch");
   }
   else if (type&0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)
            >=NAME_SIZE,1,"check_files [check5.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check5.c]","Lattice size mismatch");

      ib=blk_index(ns,bs,&nb);
      nion=iodat.nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [check5.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,
                      nbase,last,nb-1)>=NAME_SIZE,1,
            "check_files [check5.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat.nb=nb;
      iodat.ib=ib;
   }
   else if (type&0x4)
   {
      nion=iodat.nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [check5.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,
                      nbase,last,NPROC-1)>=NAME_SIZE,1,
            "check_files [check5.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void set_fld(int icnfg)
{
   int type;
   double wt1,wt2;
   char *cnfg_dir;

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();
   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;

   if (type&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file,0x0);
   }
   else if (type&0x2)
   {
      set_nio_streams(iodat.nio_streams);
      sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,iodat.ib);
      blk_import_cnfg(cnfg_file,0x0);
   }
   else
   {
      set_nio_streams(iodat.nio_streams);
      sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
      read_cnfg(cnfg_file);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Gauge field read from disk in %.2e sec\n\n",
             wt2-wt1);
      fflush(flog);
   }
}


static void Dhatop_dble(spinor_dble *s,spinor_dble *r)
{
   Dwhat_dble(mu,s,r);
   mulg5_dble(VOLUME/2,r);
   mu=-mu;
}


int main(int argc,char *argv[])
{
   int icnfg,istop,status;
   double m0,rho,nrm,del;
   double wt1,wt2,wdt;
   qflt rqsm;
   complex_dble z;
   spinor_dble **psd;
   lat_parms_t lat;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check5.in","r",stdin);

      printf("\n");
      printf("Check and performance of the CG solver\n");
      printf("--------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   read_iodat();
   read_lat_parms();
   read_bc_parms();
   read_CG_parms();

   if (my_rank==0)
      fclose(fin);

   check_machine();
   print_lat_parms();
   print_bc_parms(3);

   if (my_rank==0)
   {
      printf("Configuration storage type = ");

      if (iodat.type&0x1)
         printf("exported\n");
      else if (iodat.type&0x2)
         printf("block-exported\n");
      else
         printf("local\n");

      if (iodat.type&0x6)
         printf("Parallel configuration input: "
                "nio_nodes = %d, nio_streams = %d\n",
                iodat.nio_nodes,iodat.nio_streams);
      printf("\n");

      printf("CG parameters:\n");
      printf("mu = %.6f\n\n",mu);
      printf("nmx = %d\n",nmx);
      printf("res = %.2e\n\n",res);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   check_files();

   lat=lat_parms();
   m0=lat.m0[0];
   set_sw_parms(m0);

   alloc_ws(5);
   alloc_wsd(6);
   psd=reserve_wsd(3);

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      set_fld(icnfg);
      set_ud_phase();

      for (istop=0;istop<2;istop++)
      {
         random_sd(VOLUME,psd[0],1.0);
         bnd_sd2zero(ALL_PTS,psd[0]);
         assign_sd2sd(VOLUME,psd[0],psd[2]);

         if (istop)
            nrm=unorm_dble(VOLUME,1,psd[0]);
         else
         {
            rqsm=norm_square_dble(VOLUME,1,psd[0]);
            nrm=sqrt(rqsm.q[0]);
         }

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         rho=tmcg(nmx,istop,res,mu,psd[0],psd[1],&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         z.re=-1.0;
         z.im=0.0;
         mulc_spinor_add_dble(VOLUME,psd[2],psd[0],z);
         del=unorm_dble(VOLUME,1,psd[2]);
         error_root(del!=0.0,1,"main [check5.c]",
                    "Source field is not preserved");

         Dw_dble(mu,psd[1],psd[2]);
         mulg5_dble(VOLUME,psd[2]);
         Dw_dble(-mu,psd[2],psd[1]);
         mulg5_dble(VOLUME,psd[1]);
         mulc_spinor_add_dble(VOLUME,psd[1],psd[0],z);

         if (istop)
            del=unorm_dble(VOLUME,1,psd[1]);
         else
         {
            rqsm=norm_square_dble(VOLUME,1,psd[1]);
            del=sqrt(rqsm.q[0]);
         }

         if (my_rank==0)
         {
            printf("Solution w/o eo-preconditioning:\n");
            printf("istop = %d, status = %d\n",istop,status);
            printf("rho   = %.2e, res   = %.2e\n",rho,res);
            printf("check = %.2e, check = %.2e\n",del,del/nrm);
            printf("time = %.2e sec (total)\n",wdt);
            if (status>0)
               printf("     = %.2e usec (per point and CG iteration)",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME)));
            printf("\n\n");
            fflush(flog);
         }

         random_sd(VOLUME,psd[0],1.0);
         bnd_sd2zero(EVEN_PTS,psd[0]);
         assign_sd2sd(VOLUME/2,psd[0],psd[2]);

         if (istop)
            nrm=unorm_dble(VOLUME/2,1,psd[0]);
         else
         {
            rqsm=norm_square_dble(VOLUME/2,1,psd[0]);
            nrm=sqrt(rqsm.q[0]);
         }

         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         rho=tmcgeo(nmx,istop,res,mu,psd[0],psd[1],&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         z.re=-1.0;
         z.im=0.0;
         mulc_spinor_add_dble(VOLUME/2,psd[2],psd[0],z);
         del=unorm_dble(VOLUME/2,1,psd[2]);
         error_root(del!=0.0,1,"main [check5.c]",
                    "Source field is not preserved");

         Dhatop_dble(psd[1],psd[2]);
         Dhatop_dble(psd[2],psd[1]);
         mulc_spinor_add_dble(VOLUME/2,psd[1],psd[0],z);

         if (istop)
            del=unorm_dble(VOLUME/2,1,psd[1]);
         else
         {
            rqsm=norm_square_dble(VOLUME/2,1,psd[1]);
            del=sqrt(rqsm.q[0]);
         }

         if (my_rank==0)
         {
            printf("Solution with eo-preconditioning:\n");
            printf("istop = %d, status = %d\n",istop,status);
            printf("rho   = %.2e, res   = %.2e\n",rho,res);
            printf("check = %.2e, check = %.2e\n",del,del/nrm);
            printf("time = %.2e sec (total)\n",wdt);
            if (status>0)
               printf("     = %.2e usec (per point and CG iteration)",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME)));
            printf("\n\n");
            fflush(flog);
         }
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
