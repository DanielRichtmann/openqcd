
/*******************************************************************************
*
* File ms1.c
*
* Copyright (C) 2012-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Stochastic estimation of reweighting factors.
*
* Syntax: ms1 -i <input file> [-a [-norng]]
*
* For usage instructions see the file README.ms1.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "dfl.h"
#include "update.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
   int nrw;
   int *nfct,*nsrc;
} file_head;

static struct
{
   array_t **sqn,**lnr;
} arrays;

static struct
{
   int nc;
   qflt ***sqn,***lnr;
} data;

static struct
{
   int type,nio_nodes,nio_streams;
   int nb,ib,bs[4];
   char cnfg_dir[NAME_SIZE];
} iodat;

static int my_rank,append,norng,endian;
static int first,last,step,level,seed;
static int ipgrd[2],**rwstat=NULL,*rlxs_state=NULL,*rlxd_state=NULL;

static char line[NAME_SIZE],nbase[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char rng_file[NAME_SIZE],rng_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;

static lat_parms_t lat;
static bc_parms_t bcp;


static void alloc_data(void)
{
   int nrw,irw,*nfct,*nsrc;
   size_t n[2];
   qflt ***qp;
   array_t **ap;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;

   qp=malloc(2*nrw*sizeof(*qp));
   ap=malloc(2*nrw*sizeof(*ap));
   error((qp==NULL)||(ap==NULL),1,"alloc_data [ms1.c]",
         "Unable to allocate data arrays");
   data.sqn=qp;
   data.lnr=qp+nrw;
   arrays.sqn=ap;
   arrays.lnr=ap+nrw;

   for (irw=0;irw<nrw;irw++)
   {
      n[0]=nfct[irw];
      n[1]=2*nsrc[irw];
      arrays.sqn[irw]=alloc_array(2,n,sizeof(double),0);
      arrays.lnr[irw]=alloc_array(2,n,sizeof(double),0);
      data.sqn[irw]=(qflt**)(arrays.sqn[irw][0].a);
      data.lnr[irw]=(qflt**)(arrays.lnr[irw][0].a);
   }
}


static void write_file_head(void)
{
   int nrw,iw,*nfct;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   iw=write_parms(fdat,2*nrw,nfct,0,NULL);

   error_root(iw!=(2+2*nrw),1,"write_file_head [ms1.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int nrw,ie,*nfct;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   ie=check_parms(fdat,2*nrw,nfct,0,NULL);

   error_root(ie!=0,1,"check_file_head [ms1.c]",
              "Unexpected data file header data");
}


static void write_data(void)
{
   int nrw,irw,iw;
   stdint_t istd[1];

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);
   error_root(iw!=1,1,"write_data [ms1.c]","Incorrect write count");

   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      write_array(fdat,arrays.sqn[irw]);
      write_array(fdat,arrays.lnr[irw]);
   }
}


static int read_data(void)
{
   int nrw,irw,ir;
   stdint_t istd[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   data.nc=(int)(istd[0]);
   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      read_array(fdat,arrays.sqn[irw]);
      read_array(fdat,arrays.lnr[irw]);
   }

   return 1;
}


static void read_dirs(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Log and data directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
}


static void read_iodat(void)
{
   int type,nion,nios;
   int i,nrw,*nfct;

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

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [ms1.c]",
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
      read_line("nrw","%d",&nrw);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_iodat [ms1.c]","Improper configuration range");
      error_root(nrw<1,1,"read_iodat [ms1.c]",
                 "The number nrw or reweighting factors must be at least 1");
   }

   MPI_Bcast(line,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nrw,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat.type=type;
   strcpy(iodat.cnfg_dir,line);
   iodat.nio_nodes=nion;
   iodat.nio_streams=nios;
   iodat.bs[0]=0;
   iodat.bs[1]=0;
   iodat.bs[2]=0;
   iodat.bs[3]=0;

   nfct=malloc(2*nrw*sizeof(*nfct));
   error(nfct==NULL,1,"read_iodat [ms1.c]",
         "Unable to allocate data array");
   for (i=0;i<(2*nrw);i++)
      nfct[i]=0;
   file_head.nrw=nrw;
   file_head.nfct=nfct;
   file_head.nsrc=nfct+nrw;
}


static void setup_files(void)
{
   error(name_size("%s/%s.ms1.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms1.c]","log_dir name is too long");
   error(name_size("%s/%s.ms1.dat~",dat_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ms1.c]","dat_dir name is too long");

   sprintf(log_file,"%s/%s.ms1.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms1.par",dat_dir,nbase);
   sprintf(dat_file,"%s/%s.ms1.dat",dat_dir,nbase);
   sprintf(rng_file,"%s/%s.ms1.rng",dat_dir,nbase);
   sprintf(end_file,"%s/%s.ms1.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(rng_save,"%s~",rng_file);

   check_dir_root(log_dir);
   check_dir_root(dat_dir);
}


static void read_lat_parms(void)
{
   int nk,isw;
   double csw,*kappa;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      nk=count_tokens("kappa");
      error_root(nk<1,1,"read_lat_parms [ms1.c]",
                 "Missing hopping parameter values");
      read_line("isw","%d",&isw);
      read_line("csw","%lf",&csw);
   }

   MPI_Bcast(&nk,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   kappa=malloc(nk*sizeof(*kappa));
   error(kappa==NULL,1,"read_lat_parms [ms1.c]",
         "Unable to allocate parameter array");
   if (my_rank==0)
      read_dprms("kappa",nk,kappa);
   MPI_Bcast(kappa,nk,MPI_DOUBLE,0,MPI_COMM_WORLD);

   lat=set_lat_parms(0.0,1.0,nk,kappa,isw,csw);
   free(kappa);

   if (append)
      check_lat_parms(fdat);
   else
      write_lat_parms(fdat);
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

   if (append)
      check_bc_parms(fdat);
   else
      write_bc_parms(fdat);
}


static void read_rw_factors(void)
{
   int nrw,*nfct,*nsrc,irw,irp;
   rw_parms_t rwp;
   rat_parms_t rp;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;

   for (irw=0;irw<nrw;irw++)
   {
      read_rw_parms(irw);
      rwp=rw_parms(irw);
      nsrc[irw]=rwp.nsrc;

      if (rwp.rwfact==RWRAT)
      {
         nfct[irw]=1;
         irp=rwp.irp;
         rp=rat_parms(irp);

         if (rp.degree==0)
            read_rat_parms(irp);
      }
      else
         nfct[irw]=rwp.nfct;
   }

   if (append)
   {
      check_rw_parms(fdat);
      check_rat_parms(fdat);
   }
   else
   {
      write_rw_parms(fdat);
      write_rat_parms(fdat);
   }
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

   if (append)
      check_sap_parms(fdat);
   else
      write_sap_parms(fdat);
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

   if (append)
      check_dfl_parms(fdat);
   else
      write_dfl_parms(fdat);
}


static void read_solvers(void)
{
   int nrw,nfct,irw,ifct,isp;
   int isap,idfl;
   rw_parms_t rwp;
   solver_parms_t sp;

   nrw=file_head.nrw;
   isap=0;
   idfl=0;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);
      nfct=rwp.nfct;

      for (ifct=0;ifct<nfct;ifct++)
      {
         isp=rwp.isp[ifct];
         sp=solver_parms(isp);

         if (sp.solver==SOLVERS)
         {
            read_solver_parms(isp);
            sp=solver_parms(isp);

            if (sp.solver==SAP_GCR)
               isap=1;
            else if (sp.solver==DFL_SAP_GCR)
            {
               isap=1;
               idfl=1;
            }
         }
      }
   }

   if (append)
      check_solver_parms(fdat);
   else
      write_solver_parms(fdat);

   if (isap)
      read_sap_parms();

   if (idfl)
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

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms1.c]",
                 "Syntax: ms1 -i <input file> [-a [-norng]]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms1.c]",
                 "Machine has unknown endianness");

      append=find_opt(argc,argv,"-a");
      norng=find_opt(argc,argv,"-norng");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms1.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norng,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_iodat();
   setup_files();

   if (my_rank==0)
   {
      if (append)
         fdat=fopen(par_file,"rb");
      else
         fdat=fopen(par_file,"wb");

      error_root(fdat==NULL,1,"read_infile [ms1.c]",
                 "Unable to open parameter file");
   }

   if (my_rank==0)
   {
      if ((append==0)||(norng))
      {
         find_section("Random number generator");
         read_line("level","%d",&level);
         read_line("seed","%d",&seed);
      }
      else
      {
         level=0;
         seed=0;
      }
   }

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

   read_lat_parms();
   read_bc_parms();
   read_rw_factors();
   read_solvers();

   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save);
   }
}


static void check_old_log(int *fst,int *lst,int *stp)
{
   int ie,ic,isv,irg,lv,sd;
   int fc,lc,dc,pc;
   int np[4],bp[4];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [ms1.c]",
              "Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;
   isv=0;
   irg=0;

   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"process grid")!=NULL)
      {
         if (sscanf(line,"%dx%dx%dx%d process grid, %dx%dx%dx%d",
                    np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8)
         {
            ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                      (np[2]!=NPROC2)||(np[3]!=NPROC3));
            ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                      (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
         }
         else
            ie|=0x1;
      }
      else if (strstr(line,"fully processed")!=NULL)
      {
         pc=lc;

         if (sscanf(line,"Configuration no %d",&lc)==1)
         {
            ic+=1;
            isv=1;
         }
         else
            ie|=0x1;

         if (ic==1)
            fc=lc;
         else if (ic==2)
            dc=lc-fc;
         else if ((ic>2)&&(lc!=(pc+dc)))
            ie|=0x2;
      }
      else if (strstr(line,"Configuration no")!=NULL)
         isv=0;
      else if (norng)
      {
         if ((strstr(line,"level =")!=NULL)&&(strstr(line,"seed =")!=NULL))
         {
            if (sscanf(line,"level = %d, seed = %d",&lv,&sd)==2)
               irg|=((lv==level)&&(sd==seed));
            else
               ie|=0x1;
         }
      }
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms1.c]",
              "Incorrect read count");

   error_root((ie&0x2)!=0x0,1,"check_old_log [ms1.c]",
              "Continuation run:\n"
              "Configuration numbers are not equally spaced");

   error_root(isv==0,1,"check_old_log [ms1.c]",
              "Continuation run:\n"
              "Log file extends beyond the last configuration save");

   error_root(irg!=0,1,"check_old_log [ms1.c]",
              "Continuation run:\n"
              "Attempt to reuse previously used ranlux level and seed");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [ms1.c]",
              "Unable to open data file");

   check_file_head();

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;

   while (read_data()==1)
   {
      pc=lc;
      lc=data.nc;
      ic+=1;

      if (ic==1)
         fc=lc;
      else if (ic==2)
         dc=lc-fc;
      else if ((ic>2)&&(lc!=(pc+dc)))
         ie|=0x1;
   }

   fclose(fdat);

   error_root(ic==0,1,"check_old_dat [ms1.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [ms1.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms1.c]",
              "Configuration range is not as reported in the log file");
}


static void check_files(void)
{
   int ie,ns[4],bs[4];
   int type,nion,nb,ib,n;
   int fst,lst,stp;
   char *cnfg_dir;

   ipgrd[0]=0;
   ipgrd[1]=0;

   if (my_rank==0)
   {
      if (append)
      {
         check_old_log(&fst,&lst,&stp);
         check_old_dat(fst,lst,stp);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [ms1.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [ms1.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         ie=check_file(log_file,"r");
         ie|=check_file(dat_file,"rb");

         error_root(ie!=0,1,"check_files [ms1.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [ms1.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);
      }
   }

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;
   iodat.nb=NPROC;
   iodat.ib=0;

   if (type==0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
            "check_files [ms1.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);

      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [ms1.c]","Lattice size mismatch");
   }
   else if (type==0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,1,
            "check_files [ms1.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [ms1.c]","Lattice size mismatch");
      iodat.bs[0]=bs[0];
      iodat.bs[1]=bs[1];
      iodat.bs[2]=bs[2];
      iodat.bs[3]=bs[3];

      ib=blk_index(ns,bs,&nb);
      nion=iodat.nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [ms1.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,nbase,last,nb-1)
            >=NAME_SIZE,1,"check_files [ms1.c]","cnfg_dir name is too long");
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
      error_root(NPROC%nion!=0,1,"check_files [ms1.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,nbase,
                      last,NPROC-1)>=NAME_SIZE,1,
            "check_files [ms1.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void print_info(void)
{
   int type,isap,idfl;
   int ik,n;
   long ip;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      if (append)
         flog=freopen(log_file,"a",stdout);
      else
         flog=freopen(log_file,"w",stdout);

      error_root(flog==NULL,1,"print_info [ms1.c]","Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Measurement of reweighting factors\n");
         printf("----------------------------------\n\n");
      }

      printf("Program version %s\n",openQCD_RELEASE);

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");

      if ((ipgrd[0]!=0)&&(ipgrd[1]!=0))
         printf("Process grid and process block size changed:\n");
      else if (ipgrd[0]!=0)
         printf("Process grid changed:\n");
      else if (ipgrd[1]!=0)
         printf("Process block size changed:\n");

      if ((append==0)||(ipgrd[0]!=0)||(ipgrd[1]!=0))
      {
         printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d process block size\n\n",
                NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);
      }

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

      if (append)
      {
         printf("Random number generator:\n");

         if (norng)
            printf("level = %d, seed = %d\n\n",level,seed);
         else
         {
            printf("State of ranlxs and ranlxd reset to the\n");
            printf("last exported state\n\n");
         }
      }
      else
      {
         printf("Random number generator:\n");
         printf("level = %d, seed = %d\n\n",level,seed);

         printf("Lattice parameters:\n");

         for (ik=0;ik<lat.nk;ik++)
         {
            n=fdigits(lat.kappa[ik]);

            if (lat.nk>=11)
               printf("kappa[%2d] = %.*f\n",ik,IMAX(n,6),lat.kappa[ik]);
            else
               printf("kappa[%1d] = %.*f\n",ik,IMAX(n,6),lat.kappa[ik]);
         }

         n=fdigits(lat.csw);
         printf("isw = %d, csw = %.*f\n\n",lat.isw,IMAX(n,1),lat.csw);

         print_bc_parms(2);
         print_rw_parms();
         print_rat_parms();
         print_solver_parms(&isap,&idfl);

         if (isap)
            print_sap_parms(0);

         if (idfl)
            print_dfl_parms(0);
      }

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
      error_root(1,1,"solver_wsize [ms1.c]",
                 "Unknown or unsupported solver");
}


static void wsize(int *nws,int *nwv,int *nwvd)
{
   int nrw,nfct;
   int irw,ifct,nsds;
   int *np,*isp;
   rw_parms_t rwp;
   solver_parms_t sp;

   (*nws)=0;
   (*nwv)=0;
   (*nwvd)=0;
   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);
      nfct=rwp.nfct;
      np=rwp.np;
      isp=rwp.isp;

      for (ifct=0;ifct<nfct;ifct++)
      {
         if (rwp.rwfact==RWRAT)
         {
            sp=solver_parms(isp[ifct]);

            if (sp.solver==MSCG)
               nsds=6+2*np[ifct];
            else
               nsds=10;

            solver_wsize(isp[ifct],nsds,np[ifct],nws,nwv,nwvd);
         }
         else
         {
            nsds=4;
            solver_wsize(isp[ifct],nsds,0,nws,nwv,nwvd);
         }
      }
   }
}


static void alloc_rwstat(void)
{
   int nrw,mfct;
   int irw,ifct;
   int **pp,*p;
   rw_parms_t rwp;

   nrw=file_head.nrw;
   mfct=0;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);

      if (mfct<rwp.nfct)
         mfct=rwp.nfct;
   }

   mfct*=2;
   pp=malloc(mfct*sizeof(*pp));
   p=malloc(3*mfct*sizeof(*p));
   error((pp==NULL)||(p==NULL),1,"alloc_rwstat [ms1.c]",
         "Unable to allocate status array");
   rwstat=pp;

   for (ifct=0;ifct<mfct;ifct++)
   {
      (*pp)=p;
      pp+=1;
      p+=3;
   }
}


static void print_rwstat(int irw)
{
   int nfct,nsrc,*isp;
   int nrs,ifct,j;
   rw_parms_t rwp;
   solver_parms_t sp;

   rwp=rw_parms(irw);
   nfct=rwp.nfct;
   nsrc=rwp.nsrc;
   isp=rwp.isp;
   nrs=0;

   for (ifct=0;ifct<nfct;ifct++)
   {
      if (ifct==0)
         printf("RWF %d: status = ",irw);
      else
         printf(";");

      sp=solver_parms(isp[ifct]);

      if (sp.solver==DFL_SAP_GCR)
      {
         for (j=0;j<2;j++)
            rwstat[ifct][j]=(rwstat[ifct][j]+(nsrc/2))/nsrc;

         printf("%d,%d",rwstat[ifct][0],rwstat[ifct][1]);
         nrs+=rwstat[ifct][2];
      }
      else
      {
         rwstat[ifct][0]=(rwstat[ifct][0]+(nsrc/2))/nsrc;
         printf("%d",rwstat[ifct][0]);
      }
   }

   if (nrs)
      printf(" (no of subspace regenerations = %d)\n",nrs);
   else
      printf("\n");
}


static void set_data(int nc)
{
   int nrw,nfct,nsrc;
   int irw,ifct,isrc,isp,j;
   double mu1,mu2;
   qflt *sqn,*lnr;
   rw_parms_t rwp;

   nrw=file_head.nrw;
   data.nc=nc;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);
      set_sw_parms(sea_quark_mass(rwp.im0));
      nsrc=rwp.nsrc;
      nfct=rwp.nfct;

      for (ifct=0;ifct<(2*nfct);ifct++)
      {
         for (j=0;j<3;j++)
            rwstat[ifct][j]=0;
      }

      if (rwp.rwfact==RWRAT)
      {
         sqn=data.sqn[irw][0];
         lnr=data.lnr[irw][0];

         for (isrc=0;isrc<nsrc;isrc++)
         {
            lnr[isrc]=rwrat(rwp.irp,nfct,rwp.np,rwp.isp,sqn+isrc,rwstat+nfct);

            for (ifct=0;ifct<nfct;ifct++)
            {
               for (j=0;j<3;j++)
                  rwstat[ifct][j]+=rwstat[nfct+ifct][j];
            }
         }
      }
      else
      {
         for (ifct=0;ifct<nfct;ifct++)
         {
            sqn=data.sqn[irw][ifct];
            lnr=data.lnr[irw][ifct];

            if (ifct>0)
               mu1=rwp.mu[ifct-1];
            else
               mu1=0.0;

            mu2=rwp.mu[ifct];
            isp=rwp.isp[ifct];

            for (isrc=0;isrc<nsrc;isrc++)
            {
               if (rwp.rwfact==RWTM1)
                  lnr[isrc]=rwtm1(mu1,mu2,isp,sqn+isrc,rwstat[nfct]);
               else if (rwp.rwfact==RWTM1_EO)
                  lnr[isrc]=rwtm1eo(mu1,mu2,isp,sqn+isrc,rwstat[nfct]);
               else if (rwp.rwfact==RWTM2)
                  lnr[isrc]=rwtm2(mu1,mu2,isp,sqn+isrc,rwstat[nfct]);
               else if (rwp.rwfact==RWTM2_EO)
                  lnr[isrc]=rwtm2eo(mu1,mu2,isp,sqn+isrc,rwstat[nfct]);

               for (j=0;j<3;j++)
                  rwstat[ifct][j]+=rwstat[nfct][j];
            }
         }
      }

      if (my_rank==0)
      {
         print_rwstat(irw);

         if (rwp.rwfact==RWRAT)
            nfct=1;
         else
            nfct=rwp.nfct;

         for (ifct=0;ifct<nfct;ifct++)
         {
            lnr=data.lnr[irw][ifct];

            if (nfct==1)
               printf("RWF %d: -ln(r) = %.4e",irw,lnr[0].q[0]);
            else
               printf("RWF %d, factor %d: -ln(r) = %.4e",irw,ifct,lnr[0].q[0]);

            if (nsrc<=4)
            {
               for (isrc=1;isrc<nsrc;isrc++)
                  printf(",%.4e",lnr[isrc].q[0]);
            }
            else
            {
               printf(",%.4e,...",lnr[1].q[0]);

               for (isrc=(nsrc-2);isrc<nsrc;isrc++)
                  printf(",%.4e",lnr[isrc].q[0]);
            }

            printf("\n");
         }
      }
   }
}


static void init_rng(void)
{
   int ic;

   if (append)
   {
      if (norng)
         start_ranlux(level,seed);
      else
      {
         ic=import_ranlux(rng_file);
         error_root(ic!=(first-step),1,"init_rng [ms1.c]",
                    "Configuration number mismatch (*.rng file)");
      }
   }
   else
      start_ranlux(level,seed);
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

      error(rlxs_state==NULL,1,"save_ranlux [ms1.c]",
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
   int nc,iend,status;
   int nws,nwv,nwvd;
   double wt1,wt2,wtavg;
   dfl_parms_t dfl;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   alloc_data();
   geometry();
   check_files();
   print_info();
   init_rng();

   wsize(&nws,&nwv,&nwvd);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   alloc_rwstat();

   dfl=dfl_parms();
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
         dfl_modes(&status);
         error_root(status<0,1,"main [ms1.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status);
      }

      set_data(nc);

      if (my_rank==0)
      {
         fdat=fopen(dat_file,"ab");
         error_root(fdat==NULL,1,"main [ms1.c]",
                    "Unable to open dat file");
         write_data();
         fclose(fdat);
      }

      export_ranlux(nc,rng_file);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));
      }

      check_endflag(&iend);

      if (my_rank==0)
      {
         fflush(flog);
         copy_file(log_file,log_save);
         copy_file(dat_file,dat_save);
         copy_file(rng_file,rng_save);
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
