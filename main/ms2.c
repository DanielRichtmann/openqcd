
/*******************************************************************************
*
* File ms2.c
*
* Copyright (C) 2012, 2013, 2016-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the spectral range of the Hermitian Dirac operator.
*
* Syntax: ms2 -i <input file>
*
* For usage instructions see the file README.ms2.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
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
#include "sap.h"
#include "dfl.h"
#include "ratfcts.h"
#include "forces.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
   int type,nio_nodes,nio_streams;
   int nb,ib,bs[4];
   char cnfg_dir[NAME_SIZE];
} iodat;

static int my_rank,endian;
static int first,last,step,np_ra,np_rb;
static int *rlxs_state=NULL,*rlxd_state=NULL;
static double ar[256];

static char line[NAME_SIZE],nbase[NAME_SIZE],log_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fend=NULL;

static lat_parms_t lat;
static bc_parms_t bcp;


static void read_dirs(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Log and data directories");
      read_line("log_dir","%s",log_dir);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
}


static void read_iodat(void)
{
   int type,nion,nios;

   if (my_rank==0)
   {
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

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [ms2.c]",
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
                 "read_iodat [ms2.c]","Improper configuration range");
   }

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
   iodat.bs[0]=0;
   iodat.bs[1]=0;
   iodat.bs[2]=0;
   iodat.bs[3]=0;
}


static void setup_files(void)
{
   error(name_size("%s/%s.ms2.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms2.c]","log_dir name is too long");

   sprintf(log_file,"%s/%s.ms2.log",log_dir,nbase);
   sprintf(end_file,"%s/%s.ms2.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);

   check_dir_root(log_dir);
}


static void read_lat_parms(void)
{
   int isw;
   double kappa,csw;

   if (my_rank==0)
   {
      find_section("Dirac operator");
      read_line("kappa","%lf",&kappa);
      read_line("isw","%d",&isw);
      read_line("csw","%lf",&csw);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   lat=set_lat_parms(0.0,1.0,1,&kappa,isw,csw);
}


static void read_bc_parms(void)
{
   int bc;
   double cF,cF_prime;
   double phi[2],phi_prime[2],theta[3];

   if (my_rank==0)
   {
      find_section("Boundary conditions");
      read_line("type","%d",&bc);

      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;
      cF=1.0;
      cF_prime=1.0;

      if (bc==1)
         read_dprms("phi",2,phi);

      if ((bc==1)||(bc==2))
         read_dprms("phi'",2,phi_prime);

      if (bc!=3)
         read_line("cF","%lf",&cF);

      if (bc==2)
         read_line("cF'","%lf",&cF_prime);

      read_dprms("theta",3,theta);
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(theta,3,MPI_DOUBLE,0,MPI_COMM_WORLD);

   bcp=set_bc_parms(bc,1.0,1.0,cF,cF_prime,phi,phi_prime,theta);
}


static void read_sap_parms(void)
{
   int bs[4];

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);
}


static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mu,res;

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
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);
}


static void read_solver(void)
{
   solver_parms_t sp;

   read_solver_parms(0);
   sp=solver_parms(0);

   if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      read_sap_parms();

   if (sp.solver==DFL_SAP_GCR)
      read_dfl_parms();
}


static void read_infile(int argc,char *argv[])
{
   int ifile;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms2.c]",
                 "Syntax: ms2 -i <input file>");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms2.c]",
                 "Machine has unknown endianness");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms2.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat();
   setup_files();
   read_lat_parms();
   read_bc_parms();

   if (my_rank==0)
   {
      find_section("Power method");
      read_line("np_ra","%d",&np_ra);
      read_line("np_rb","%d",&np_rb);
      error_root((np_ra<1)||(np_rb<1),1,"read_infile [ms2.c]",
                 "Power method iteration numbers must be at least 1");
   }

   MPI_Bcast(&np_ra,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&np_rb,1,MPI_INT,0,MPI_COMM_WORLD);
   read_solver();

   if (my_rank==0)
      fclose(fin);
}


static void check_files(void)
{
   int ie,ns[4],bs[4];
   int type,nion,nb,ib,n;
   char *cnfg_dir;

   if (my_rank==0)
   {
      ie=check_file(log_file,"r");
      error_root(ie!=0,1,"check_files [ms2.c]",
                 "Attempt to overwrite old *.log file");
   }

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;
   iodat.nb=NPROC;
   iodat.ib=0;

   if (type==0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
            "check_files [ms2.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);
      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);

      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [ms2.c]","Lattice size mismatch");
   }
   else if (type==0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,1,
            "check_files [ms2.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [ms2.c]","Lattice size mismatch");
      iodat.bs[0]=bs[0];
      iodat.bs[1]=bs[1];
      iodat.bs[2]=bs[2];
      iodat.bs[3]=bs[3];

      ib=blk_index(ns,bs,&nb);
      nion=iodat.nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [ms2.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,nbase,last,nb-1)
            >=NAME_SIZE,1,"check_files [ms2.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat.nb=nb;
      iodat.ib=ib;
   }
   else
   {
      nion=iodat.nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [ms2.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,nbase,
                      last,NPROC-1)>=NAME_SIZE,1,
                 "check_files [ms2.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void print_info(void)
{
   int isap,idfl,type,n;
   long ip;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      flog=freopen(log_file,"w",stdout);
      error_root(flog==NULL,1,"print_info [ms2.c]","Unable to open log file");
      printf("\n");

      printf("Spectral range of the Hermitian Dirac operator\n");
      printf("----------------------------------------------\n\n");

      printf("Program version %s\n",openQCD_RELEASE);

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");

      printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
      printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d process block size\n\n",
             NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);

      printf("Configuration storage type = ");
      type=iodat.type;

      if (type==0x1)
         printf("exported\n");
      else if (type==0x2)
         printf("block-exported\n");
      else
         printf("local\n");

      if (type&0x6)
      {
         printf("Parallel I/O parameters: "
                "nio_nodes = %d, nio_streams = %d\n",
                iodat.nio_nodes,iodat.nio_streams);
      }

      if (type==0x2)
      {
         printf("Block size = %dx%dx%dx%d\n",iodat.bs[0],
                iodat.bs[1],iodat.bs[2],iodat.bs[3]);
      }

      printf("\n");
      printf("Dirac operator:\n");
      n=fdigits(lat.kappa[0]);
      printf("kappa = %.*f\n",IMAX(n,6),lat.kappa[0]);
      n=fdigits(lat.csw);
      printf("isw = %d, csw = %.*f\n\n",lat.isw,IMAX(n,1),lat.csw);
      print_bc_parms(2);

      printf("Power method:\n");
      printf("np_ra = %d\n",np_ra);
      printf("np_rb = %d\n\n",np_rb);

      print_bc_parms(2);
      print_solver_parms(&isap,&idfl);

      if (isap)
         print_sap_parms(0);

      if (idfl)
         print_dfl_parms(0);

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);
      fflush(flog);
   }
}


static void maxn(int *n,int m)
{
   if ((*n)<m)
      (*n)=m;
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();

   maxn(nws,dp.Ns+2);
   maxn(nwv,2*dpp.nkv+2);
   maxn(nwvd,4);
}


static void solver_wsize(int isp,int nsds,int np,int *nws,int *nwv,int *nwvd)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if (sp.solver==CGNE)
      maxn(nws,nsds+11);
   else if (sp.solver==MSCG)
   {
      if (np>1)
         maxn(nws,nsds+2*np+6);
      else
         maxn(nws,nsds+10);
   }
   else if (sp.solver==SAP_GCR)
      maxn(nws,nsds+2*sp.nkv+5);
   else if (sp.solver==DFL_SAP_GCR)
   {
      maxn(nws,nsds+2*sp.nkv+6);
      dfl_wsize(nws,nwv,nwvd);
   }
   else
      error_root(1,1,"solver_wsize [ms2.c]",
                 "Unknown or unsupported solver");
}


static void wsize(int *nws,int *nwv,int *nwvd)
{
   int nsds;

   (*nws)=0;
   (*nwv)=0;
   (*nwvd)=0;

   nsds=2;
   solver_wsize(0,nsds,0,nws,nwv,nwvd);

   if ((*nws)<4)
      (*nws)=4;
}


static double power1(int *status)
{
   int k,l,stat[6];
   double r;
   spinor_dble *phi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;

   set_sw_parms(sea_quark_mass(0));
   sp=solver_parms(0);

   if (sp.solver==DFL_SAP_GCR)
   {
      for (l=0;l<3;l++)
         status[l]=0;
   }
   else
      status[0]=0;

   if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
   }

   wsd=reserve_wsd(1);
   phi=wsd[0];
   random_sd(VOLUME/2,phi,1.0);
   bnd_sd2zero(EVEN_PTS,phi);
   r=normalize_dble(VOLUME/2,1,phi);

   for (k=0;k<np_ra;k++)
   {
      if (sp.solver==CGNE)
      {
         tmcgeo(sp.nmx,sp.istop,sp.res,0.0,phi,phi,stat);

         error_root(stat[0]<0,1,"power1 [ms2.c]",
                    "CGNE solver failed (status = %d)",stat[0]);

         if (status[0]<stat[0])
            status[0]=stat[0];
      }
      else if (sp.solver==SAP_GCR)
      {
         mulg5_dble(VOLUME/2,phi);
         set_sd2zero(VOLUME/2,phi+(VOLUME/2));
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,stat);
         mulg5_dble(VOLUME/2,phi);
         set_sd2zero(VOLUME/2,phi+(VOLUME/2));
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,stat+1);

         error_root((stat[0]<0)||(stat[1]<0),1,"power1 [ms2.c]",
                    "SAP_GCR solver failed (status = %d;%d)",
                    stat[0],stat[1]);

         for (l=0;l<2;l++)
         {
            if (status[0]<stat[l])
               status[0]=stat[l];
         }
      }
      else if (sp.solver==DFL_SAP_GCR)
      {
         mulg5_dble(VOLUME/2,phi);
         set_sd2zero(VOLUME/2,phi+(VOLUME/2));
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,stat);
         mulg5_dble(VOLUME/2,phi);
         set_sd2zero(VOLUME/2,phi+(VOLUME/2));
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,0.0,phi,phi,stat+3);

         error_root((stat[0]<0)||(stat[1]<0)||(stat[3]<0)||(stat[4]<0),1,
                    "power1 [ms2.c]","DFL_SAP_GCR solver failed "
                    "(status = %d,%d,%d;%d,%d,%d)",
                    stat[0],stat[1],stat[2],stat[3],stat[4],stat[5]);

         for (l=0;l<2;l++)
         {
            if (status[l]<stat[l])
               status[l]=stat[l];

            if (status[l]<stat[l+3])
               status[l]=stat[l+3];
         }

         status[2]+=(stat[2]!=0);
         status[2]+=(stat[5]!=0);
      }

      r=normalize_dble(VOLUME/2,1,phi);
   }

   release_wsd();

   return 1.0/sqrt(r);
}


static double power2(void)
{
   int k;
   double r;
   spinor_dble *phi,*psi,**wsd;

   set_sw_parms(sea_quark_mass(0));
   sw_term(ODD_PTS);

   wsd=reserve_wsd(2);
   phi=wsd[0];
   psi=wsd[1];

   random_sd(VOLUME/2,phi,1.0);
   bnd_sd2zero(EVEN_PTS,phi);
   r=normalize_dble(VOLUME/2,1,phi);

   for (k=0;k<np_rb;k++)
   {
      Dwhat_dble(0.0,phi,psi);
      mulg5_dble(VOLUME/2,psi);
      Dwhat_dble(0.0,psi,phi);
      mulg5_dble(VOLUME/2,phi);

      r=normalize_dble(VOLUME/2,1,phi);
   }

   release_wsd();

   return sqrt(r);
}


static void save_ranlux(void)
{
   int nlxs,nlxd;

   if (rlxs_state==NULL)
   {
      nlxs=rlxs_size();
      nlxd=rlxd_size();

      rlxs_state=malloc((nlxs+nlxd)*sizeof(int));
      rlxd_state=rlxs_state+nlxs;

      error(rlxs_state==NULL,1,"save_ranlux [ms2.c]",
            "Unable to allocate state arrays");
   }

   rlxs_get(rlxs_state);
   rlxd_get(rlxd_state);
}


static void restore_ranlux(void)
{
   rlxs_reset(rlxs_state);
   rlxd_reset(rlxd_state);
}


static void read_ud(int icnfg)
{
   int type,nios,ib;
   double wt1,wt2;
   char *cnfg_dir;

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;
   nios=iodat.nio_streams;
   ib=iodat.ib;

   if (type==0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file,0x0);
   }
   else if (type==0x2)
   {
      set_nio_streams(nios);
      sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,ib);
      blk_import_cnfg(cnfg_file,0x0);
   }
   else
   {
      save_ranlux();
      set_nio_streams(nios);
      sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
      read_cnfg(cnfg_file);
      restore_ranlux();
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Configuration read from disk in %.2e sec\n\n",
             wt2-wt1);
      fflush(flog);
   }
}


static void check_endflag(int *iend)
{
   if (my_rank==0)
   {
      fend=fopen(end_file,"r");

      if (fend!=NULL)
      {
         fclose(fend);
         remove(end_file);
         (*iend)=1;
         printf("End flag set, run stopped\n\n");
      }
      else
         (*iend)=0;
   }

   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
   int nc,iend,status[3];
   int nws,nwv,nwvd,n,bc;
   double ra,ramin,ramax,raavg;
   double rb,rbmin,rbmax,rbavg;
   double A,eps,delta,Ne,d1,d2;
   double wt1,wt2,wtavg;
   dfl_parms_t dfl;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   geometry();
   check_files();
   print_info();
   dfl=dfl_parms();
   start_ranlux(0,1234);

   wsize(&nws,&nwv,&nwvd);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);

   ramin=0.0;
   ramax=0.0;
   raavg=0.0;

   rbmin=0.0;
   rbmax=0.0;
   rbavg=0.0;

   iend=0;
   wtavg=0.0;

   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      if (my_rank==0)
      {
         printf("Configuration no %d\n",nc);
         fflush(flog);
      }

      read_ud(nc);
      set_ud_phase();

      if (dfl.Ns)
      {
         dfl_modes(status);
         error_root(status[0]<0,1,"main [ms2.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);
      }

      ra=power1(status);
      rb=power2();

      if (nc==first)
      {
         ramin=ra;
         ramax=ra;
         raavg=ra;

         rbmin=rb;
         rbmax=rb;
         rbavg=rb;
      }
      else
      {
         if (ra<ramin)
            ramin=ra;
         if (ra>ramax)
            ramax=ra;
         raavg+=ra;

         if (rb<rbmin)
            rbmin=rb;
         if (rb>rbmax)
            rbmax=rb;
         rbavg+=rb;
      }

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         printf("ra = %.2e, rb = %.2e, ",ra,rb);

         if (dfl.Ns)
            printf("status = %d,%d,%d\n",
                   status[0],status[1],status[2]);
         else
            printf("status = %d\n",status[0]);

         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));

         fflush(flog);
         copy_file(log_file,log_save);
      }

      check_endflag(&iend);
   }

   if (my_rank==0)
   {
      last=nc-step;
      nc=(last-first)/step+1;

      printf("Summary\n");
      printf("-------\n\n");

      printf("Considered %d configurations in the range %d -> %d\n\n",
             nc,first,last);

      printf("The three figures quoted in each case are the minimal,\n");
      printf("maximal and average values\n\n");

      printf("Spectral gap ra    = %.2e, %.2e, %.2e\n",
             ramin,ramax,raavg/(double)(nc));
      printf("Spectral radius rb = %.2e, %.2e, %.2e\n\n",
             rbmin,rbmax,rbavg/(double)(nc));

      ra=0.90*ramin;
      rb=1.10*rbmax;
      eps=ra/rb;
      eps=eps*eps;

      bc=bc_type();
      Ne=0.5*(double)(N1*N2*N3);

      if (bc==0)
         Ne*=(double)(N0-2);
      else if ((bc==1)||(bc==2))
         Ne*=(double)(N0-1);
      else
         Ne*=(double)(N0);

      printf("Zolotarev rational approximation:\n\n");

      printf("n: number of poles\n");
      printf("delta: approximation error\n");
      printf("Ne: number of even lattice points\n");
      printf("Suggested spectral range = [%.2e,%.2e]\n\n",ra,rb);

      printf("     n      delta    12*Ne*delta     12*Ne*delta^2\n");

      for (n=6;n<=128;n++)
      {
         zolotarev(n,eps,&A,ar,&delta);
         d1=12.0*Ne*delta;
         d2=d1*delta;

         printf("   %3d     %.1e      %.1e         %.1e\n",n,delta,d1,d2);

         if ((d1<1.0e-2)&&(d2<1.0e-4))
            break;
      }

      printf("\n");
   }

   if (my_rank==0)
   {
      fflush(flog);
      copy_file(log_file,log_save);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
