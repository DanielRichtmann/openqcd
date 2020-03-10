
/*******************************************************************************
*
* File ms4.c
*
* Copyright (C) 2012, 2013, 2016-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of quark propagators.
*
* Syntax: ms4 -i <input file>
*
* For usage instructions see the file README.ms4.
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
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
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
} iodat[2];

static int my_rank,endian;
static int first,last,step;
static int level,seed,x0,nsrc;
static int *rlxs_state=NULL,*rlxd_state=NULL;
static double mus;

static char line[NAME_SIZE],nbase[NAME_SIZE];
static char log_dir[NAME_SIZE],end_file[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE];
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


static void read_iodat0(void)
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

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat0 [ms4.c]",
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
                 "read_iodat0 [ms4.c]","Improper configuration range");
   }

   MPI_Bcast(line,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat[0].type=type;
   strcpy(iodat[0].cnfg_dir,line);
   iodat[0].nio_nodes=nion;
   iodat[0].nio_streams=nios;
   iodat[0].bs[0]=0;
   iodat[0].bs[1]=0;
   iodat[0].bs[2]=0;
   iodat[0].bs[3]=0;
}


static void read_iodat1(void)
{
   int type,nion,nios,bs[4];

   if (my_rank==0)
   {
      find_section("Propagators");

      read_line("type","%s",line);

      if (strchr(line,'e')!=NULL)
         type=0x1;
      else if (strchr(line,'b')!=NULL)
         type=0x2;
      else if (strchr(line,'l')!=NULL)
         type=0x4;
      else
         type=0x0;

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat1 [ms4.c]",
                 "Improper propagator storage type");

      read_line("prop_dir","%s",line);

      if (type==0x2)
         read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      else
      {
         bs[0]=0;
         bs[1]=0;
         bs[2]=0;
         bs[3]=0;
      }

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
   }

   MPI_Bcast(line,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);

   iodat[1].type=type;
   strcpy(iodat[1].cnfg_dir,line);
   iodat[1].nio_nodes=nion;
   iodat[1].nio_streams=nios;
   iodat[1].bs[0]=bs[0];
   iodat[1].bs[1]=bs[1];
   iodat[1].bs[2]=bs[2];
   iodat[1].bs[3]=bs[3];
}


static void setup_files(void)
{
   check_dir_root(log_dir);
   error(name_size("%s/%s.ms4.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms4.c]","log_dir name is too long");

   sprintf(log_file,"%s/%s.ms4.log",log_dir,nbase);
   sprintf(end_file,"%s/%s.ms4.end",log_dir,nbase);
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
      read_line("mu","%lf",&mus);
      read_line("isw","%d",&isw);
      read_line("csw","%lf",&csw);

      find_section("Source fields");
      read_line("x0","%d",&x0);
      read_line("nsrc","%d",&nsrc);

      error_root((x0<0)||(x0>=N0),1,"read_lat_parms [ms4.c]",
                 "Specified time x0 is out of range");
      error_root(nsrc<1,1,"read_lat_parms [ms4.c]",
                 "The number of source fields must be at least 1");
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mus,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   MPI_Bcast(&x0,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nsrc,1,MPI_INT,0,MPI_COMM_WORLD);

   lat=set_lat_parms(0.0,1.0,1,&kappa,isw,csw);
   set_sw_parms(sea_quark_mass(0));
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

      error_root(((x0==0)&&(bc!=3))||((x0==(N0-1))&&(bc==0)),1,
                 "read_bc_parms [ms4.c]","Incompatible choice of boundary "
                 "conditions and source time");

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

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms4.c]",
                 "Syntax: ms4 -i <input file>");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms4.c]",
                 "Machine has unknown endianness");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms4.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat0();
   read_iodat1();

   if (my_rank==0)
   {
      find_section("Random number generator");
      read_line("level","%d",&level);
      read_line("seed","%d",&seed);
   }

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

   setup_files();
   read_lat_parms();
   read_bc_parms();
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
      error_root(ie!=0,1,"check_files [ms4.c]",
                 "Attempt to overwrite old *.log file");
   }

   type=iodat[0].type;
   cnfg_dir=iodat[0].cnfg_dir;
   iodat[0].nb=NPROC;
   iodat[0].ib=0;

   if (type==0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
            "check_files [ms4.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);

      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [ms4.c]","Lattice size mismatch");
   }
   else if (type==0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,1,
            "check_files [ms4.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [ms4.c]","Lattice size mismatch");
      iodat[0].bs[0]=bs[0];
      iodat[0].bs[1]=bs[1];
      iodat[0].bs[2]=bs[2];
      iodat[0].bs[3]=bs[3];

      ib=blk_index(ns,bs,&nb);
      nion=iodat[0].nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [ms4.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,
                      nbase,last,nb-1)>=NAME_SIZE,1,
            "check_files [ms4.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat[0].nb=nb;
      iodat[0].ib=ib;
   }
   else
   {
      nion=iodat[0].nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [ms4.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,nbase,
                      last,NPROC-1)>=NAME_SIZE,1,
            "check_files [ms4.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }

   type=iodat[1].type;
   cnfg_dir=iodat[1].cnfg_dir;
   iodat[1].nb=NPROC;
   iodat[1].ib=0;

   if (type==0x1)
   {
      error(name_size("%s/%sn%d.s%d",cnfg_dir,nbase,last,nsrc-1)>=NAME_SIZE,1,
            "check_files [ms4.c]","prop_dir name is too long");
      check_dir_root(cnfg_dir);
   }
   else if (type==0x2)
   {
      ns[0]=N0;
      ns[1]=N1;
      ns[2]=N2;
      ns[3]=N3;
      bs[0]=iodat[1].bs[0];
      bs[1]=iodat[1].bs[1];
      bs[2]=iodat[1].bs[2];
      bs[3]=iodat[1].bs[3];

      ib=blk_index(ns,bs,&nb);
      nion=iodat[1].nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [ms4.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d.s%d_b%d",cnfg_dir,nion-1,n-1,
                      nbase,last,nsrc-1,nb-1)>=NAME_SIZE,1,
            "check_files [ms4.c]","flds_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat[1].nb=nb;
      iodat[1].ib=ib;
   }
   else
   {
      nion=iodat[1].nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [ms4.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d.s%d_%d",cnfg_dir,nion-1,n-1,nbase,
                      last,nsrc-1,NPROC-1)>=NAME_SIZE,1,
            "check_files [ms4.c]","flds_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void print_info(void)
{
   int type,isap,idfl;
   int n,i;
   long ip;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      flog=freopen(log_file,"w",stdout);
      error_root(flog==NULL,1,"print_info [ms4.c]","Unable to open log file");
      printf("\n");

      printf("Computation of quark propagators\n");
      printf("--------------------------------\n\n");

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

      for (i=0;i<2;i++)
      {
         if (i==0)
            printf("Configuration storage type = ");
         else
            printf("Propagator storage type = ");

         type=iodat[i].type;

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
                   iodat[i].nio_nodes,iodat[i].nio_streams);
         }

         if (type==0x2)
         {
            printf("Block size = %dx%dx%dx%d\n",iodat[i].bs[0],
                   iodat[i].bs[1],iodat[i].bs[2],iodat[i].bs[3]);
         }
      }

      printf("\n");
      printf("Random number generator:\n");
      printf("level = %d, seed = %d\n\n",level,seed);

      printf("Dirac operator:\n");
      n=fdigits(lat.kappa[0]);
      printf("kappa = %.*f\n",IMAX(n,6),lat.kappa[0]);
      n=fdigits(mus);
      printf("mu = %.*f\n",IMAX(n,1),mus);
      n=fdigits(lat.csw);
      printf("isw = %d, csw = %.*f\n\n",lat.isw,IMAX(n,1),lat.csw);
      print_bc_parms(2);

      printf("Source fields:\n");
      printf("x0 = %d\n",x0);
      printf("nsrc = %d\n\n",nsrc);

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

   nsds=4;
   solver_wsize(0,nsds,0,nws,nwv,nwvd);

   if ((*nws)<4)
      (*nws)=4;
}


static void random_source(spinor_dble *eta)
{
   int y0,iy,ix;

   set_sd2zero(VOLUME,eta);
   y0=x0-cpr[0]*L0;

   if ((y0>=0)&&(y0<L0))
   {
      for (iy=0;iy<(L1*L2*L3);iy++)
      {
         ix=ipt[iy+y0*L1*L2*L3];
         random_sd(1,eta+ix,1.0);
      }
   }
}


static void solve_dirac(spinor_dble *eta,spinor_dble *psi,int *status)
{
   solver_parms_t sp;
   sap_parms_t sap;

   sp=solver_parms(0);

   if (sp.solver==CGNE)
   {
      mulg5_dble(VOLUME,eta);

      tmcg(sp.nmx,sp.istop,sp.res,mus,eta,eta,status);

      error_root(status[0]<0,1,"solve_dirac [ms4.c]",
                 "CGNE solver failed (status = %d)",status[0]);

      Dw_dble(-mus,eta,psi);
      mulg5_dble(VOLUME,psi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mus,eta,psi,status);

      error_root(status[0]<0,1,"solve_dirac [ms4.c]",
                 "SAP_GCR solver failed (status = %d)",status[0]);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mus,eta,psi,status);

      error_root((status[0]<0)||(status[1]<0),1,
                 "solve_dirac [ms4.c]","DFL_SAP_GCR solver failed "
                 "(status = %d,%d,%d)",status[0],status[1],status[2]);
   }
   else
      error_root(1,1,"solve_dirac [ms4.c]",
                 "Unknown or unsupported solver");
}



static void save_prop(int icnfg,int isrc,spinor_dble *sd)
{
   int type;
   char *cnfg_dir;

   type=iodat[1].type;
   cnfg_dir=iodat[1].cnfg_dir;

   if (type==0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d.s%d",
              cnfg_dir,nbase,icnfg,isrc);
      export_sfld(cnfg_file,0,sd);
   }
   else if (type==0x2)
   {
      sprintf(cnfg_file,"%s/%sn%d.s%d_b%d",
              cnfg_dir,nbase,icnfg,isrc,iodat[1].ib);
      blk_export_sfld(iodat[1].bs,cnfg_file,0,sd);
   }
   else if (type==0x4)
   {
      sprintf(cnfg_file,"%s/%sn%d.s%d_%d",
              cnfg_dir,nbase,icnfg,isrc,my_rank);
      write_sfld(cnfg_file,0,sd);
   }
}


static void propagator(int nc,int *status,double *wtsum)
{
   int isrc,l,stat[3];
   double wt[2];
   spinor_dble *eta,*psi,**wsd;

   wsd=reserve_wsd(2);
   eta=wsd[0];
   psi=wsd[1];
   wtsum[0]=0.0;
   wtsum[1]=0.0;

   for (l=0;l<3;l++)
   {
      status[l]=0;
      stat[l]=0;
   }

   for (isrc=0;isrc<nsrc;isrc++)
   {
      random_source(eta);

      MPI_Barrier(MPI_COMM_WORLD);
      wt[0]=MPI_Wtime();

      solve_dirac(eta,psi,stat);

      MPI_Barrier(MPI_COMM_WORLD);
      wt[1]=MPI_Wtime();
      wtsum[0]+=(wt[1]-wt[0]);

      for (l=0;l<2;l++)
         status[l]+=stat[l];

      status[2]+=(stat[2]!=0);

      save_prop(nc,isrc,psi);
      MPI_Barrier(MPI_COMM_WORLD);
      wt[0]=MPI_Wtime();
      wtsum[1]+=(wt[0]-wt[1]);
   }

   for (l=0;l<2;l++)
      status[l]=(status[l]+(nsrc/2))/nsrc;

   wtsum[0]/=(double)(nsrc);
   wtsum[1]/=(double)(nsrc);

   release_wsd();
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

      error(rlxs_state==NULL,1,"save_ranlux [ms4.c]",
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

   type=iodat[0].type;
   cnfg_dir=iodat[0].cnfg_dir;
   nios=iodat[0].nio_streams;
   ib=iodat[0].ib;

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
   int nws,nwv,nwvd,n;
   double wt[2],wtavg[2];
   dfl_parms_t dfl;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   geometry();
   check_files();
   print_info();
   dfl=dfl_parms();
   start_ranlux(level,seed);

   wsize(&nws,&nwv,&nwvd);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);

   iend=0;
   wtavg[0]=0.0;
   wtavg[1]=0.0;

   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
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
         error_root(status[0]<0,1,"main [ms4.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);

         if (my_rank==0)
            printf("Deflation subspace generation: status = %d\n",status[0]);
      }

      propagator(nc,status,wt);
      wtavg[0]+=wt[0];
      wtavg[1]+=wt[1];

      if (my_rank==0)
      {
         printf("Computation of propagator completed\n");

         if (dfl.Ns)
         {
            printf("status = %d,%d",status[0],status[1]);

            if (status[2])
               printf(" (no of subspace regenerations = %d)\n",status[2]);
            else
               printf("\n");
         }
         else
            printf("status = %d\n",status[0]);

         n=(nc-first)/step+1;

         printf("Dirac equation solved in %.2e sec per source field "
                "(average %.2e sec)\n",wt[0],wtavg[0]/(double)(n));
         printf("Solution saved in %.2e sec (average %.2e sec)\n",
                wt[1],wtavg[1]/(double)(n));
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,(double)(nsrc)*(wt[0]+wt[1]));
         printf("(average = %.2e sec)\n\n",
                (double)(nsrc)*(wtavg[0]+wtavg[1])/(double)(n));

         fflush(flog);
         copy_file(log_file,log_save);
      }

      check_endflag(&iend);
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
