
/*******************************************************************************
*
* File ms1.c
*
* Copyright (C) 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Measurement of reweighting factors
*
* Syntax: ms1 -i <input file> [-noexp] [-a]
*
* For usage instructions see the file README.ms1
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

#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

static struct
{
   int nrw;
   int *nsrc;
} file_head;

static struct
{
   int nc;
   double **sqn,**lnr;
} data;

static int my_rank,noexp,append,endian;
static int first,last,step,level,seed;
static int **rwstat=NULL,*rlxs_state=NULL,*rlxd_state=NULL;

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],cnfg_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void alloc_data(void)
{
   int nrw,*nsrc;
   int n,l;
   double **pp,*p;

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   n=0;

   for (l=0;l<nrw;l++)
      n+=nsrc[l];

   pp=malloc(2*nrw*sizeof(*pp));
   p=malloc(2*n*sizeof(*p));
   error((pp==NULL)||(p==NULL),1,"alloc_data [ms1.c]",
         "Unable to allocate data arrays");

   data.sqn=pp;
   data.lnr=pp+nrw;
   
   for (l=0;l<nrw;l++)
   {
      data.sqn[l]=p;
      p+=nsrc[l];
      data.lnr[l]=p;
      p+=nsrc[l];
   }
}


static void write_file_head(void)
{
   int nrw,*nsrc;
   int iw,l;
   stdint_t istd[1];

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   
   istd[0]=(stdint_t)(nrw);
   
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   for (l=0;l<nrw;l++)
   {
      istd[0]=(stdint_t)(nsrc[l]);
   
      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      iw+=fwrite(istd,sizeof(stdint_t),1,fdat);
   }

   error_root(iw!=(1+nrw),1,"write_file_head [ms1.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int nrw,*nsrc;
   int ir,ie,l;
   stdint_t istd[1];

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   ie=(istd[0]!=(stdint_t)(nrw));

   for (l=0;l<nrw;l++)
   {
      ir+=fread(istd,sizeof(stdint_t),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      ie|=(istd[0]!=(stdint_t)(nsrc[l]));
   }
   
   error_root(ir!=(1+nrw),1,"check_file_head [ms1.c]",
              "Incorrect read count");
   
   error_root(ie!=0,1,"check_file_head [ms1.c]",
              "Unexpected value of nrw or nsrc");
}


static void write_data(void)
{
   int iw,n;
   int nrw,*nsrc,irw,isrc;
   stdint_t istd[1];
   double dstd[1];   

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   n=0;

   for (irw=0;irw<nrw;irw++)
   {
      for (isrc=0;isrc<nsrc[irw];isrc++)
      {
         dstd[0]=data.sqn[irw][isrc];

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         iw+=fwrite(dstd,sizeof(double),1,fdat);
      }

      for (isrc=0;isrc<nsrc[irw];isrc++)
      {
         dstd[0]=data.lnr[irw][isrc];

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         iw+=fwrite(dstd,sizeof(double),1,fdat);
      }

      n+=nsrc[irw];
   }
   
   error_root(iw!=(1+2*n),1,"write_data [ms1.c]",
              "Incorrect write count");
}


static int read_data(void)
{
   int ir,n;
   int nrw,*nsrc,irw,isrc;
   stdint_t istd[1];
   double dstd[1];
   
   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   
   data.nc=(int)(istd[0]);

   nrw=file_head.nrw;      
   nsrc=file_head.nsrc;
   n=0;

   for (irw=0;irw<nrw;irw++)
   {
      for (isrc=0;isrc<nsrc[irw];isrc++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
            
         data.sqn[irw][isrc]=dstd[0];
      }

      for (isrc=0;isrc<nsrc[irw];isrc++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
            
         data.lnr[irw][isrc]=dstd[0];
      }

      n+=nsrc[irw];
   }

   error_root(ir!=(1+2*n),1,"read_data [ms1.c]",
              "Read error or incomplete data record");

   return 1;
}


static void read_dirs(void)
{
   int nrw,*nsrc;
   
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);

      if (noexp)
      {
         read_line("loc_dir","%s",loc_dir);
         cnfg_dir[0]='\0';
      }
      else
      {
         read_line("cnfg_dir","%s",cnfg_dir);         
         loc_dir[0]='\0';
      }

      find_section("Configurations");
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);
      read_line("nrw","%d",&nrw);
      
      find_section("Random number generator");
      read_line("level","%d",&level);
      read_line("seed","%d",&seed);     

      error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                 "read_dirs [ms1.c]","Improper configuration range");
      error_root(nrw<1,1,"read_dirs [ms1.c]",
                 "The number nrw or reweighting factors must be at least 1");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nrw,1,MPI_INT,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

   nsrc=malloc(nrw*sizeof(*nsrc));
   error(nsrc==NULL,1,"read_dirs [ms1.c]",
         "Unable to allocate data array");
   file_head.nrw=nrw;
   file_head.nsrc=nsrc;   
}


static void setup_files(void)
{
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [ms1.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [ms1.c]","cnfg_dir name is too long");

   check_dir_root(log_dir);
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.ms1.log~",log_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms1.c]","log_dir name is too long");
   error_root(name_size("%s/%s.ms1.dat~",dat_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms1.c]","dat_dir name is too long");   
      
   sprintf(log_file,"%s/%s.ms1.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms1.par",dat_dir,nbase);   
   sprintf(dat_file,"%s/%s.ms1.dat",dat_dir,nbase);
   sprintf(end_file,"%s/%s.ms1.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);   
   sprintf(dat_save,"%s~",dat_file);
}


static void read_lat_parms(void)
{
   double kappa_u,kappa_s,kappa_c,csw,cF;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("kappa_u","%lf",&kappa_u);
      read_line("kappa_s","%lf",&kappa_s);
      read_line("kappa_c","%lf",&kappa_c);      
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);   
   }

   MPI_Bcast(&kappa_u,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&kappa_s,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&kappa_c,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   set_lat_parms(0.0,1.0,kappa_u,kappa_s,kappa_c,csw,1.0,cF);

   if (append)
      check_lat_parms(fdat);
   else
      write_lat_parms(fdat);
}


static void read_rw_factors(void)
{
   int nrw,*nsrc,irw,irp;
   rw_parms_t rwp;
   rat_parms_t rp;

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;

   for (irw=0;irw<nrw;irw++)
   {
      read_rw_parms(irw);
      rwp=rw_parms(irw);
      nsrc[irw]=rwp.nsrc;

      if (rwp.rwfact==RWRAT)
      {
         irp=rwp.irp;
         rp=rat_parms(irp);

         if (rp.degree==0)
            read_rat_parms(irp);
      }
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
   double kappa,mu,res,resd;

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
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy,nkv,nmx,res);
   
   if (my_rank==0)
   {
      find_section("Deflation projectors");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("resd","%lf",&resd);
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&resd,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,resd,res);
   
   if (append)
      check_dfl_parms(fdat);
   else
      write_dfl_parms(fdat);
}


static void read_solvers(void)
{
   int nrw,irw;
   int n,l,j;
   int isap,idfl;
   rw_parms_t rwp;
   solver_parms_t sp;

   nrw=file_head.nrw;
   isap=0;
   idfl=0;
   
   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);
      n=rwp.n;

      for (l=0;l<n;l++)
      {         
         j=rwp.isp[l];
         sp=solver_parms(j);

         if (sp.solver==SOLVERS)
         {
            read_solver_parms(j);
            sp=solver_parms(j);

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
                 "Syntax: ms1 -i <input file> [-noexp] [-a]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms1.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp");      
      append=find_opt(argc,argv,"-a");
      
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms1.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   
   read_dirs();
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

   read_lat_parms();
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
   int ie,ic,isv;
   int fc,lc,dc,pc;
   
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
         
   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"fully processed")!=NULL)
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
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms1.c]",
              "Incorrect read count");   
   error_root((ie&0x2)!=0x0,1,"check_old_log [ms1.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [ms1.c]",
              "Log file extends beyond the last configuration save");

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
   int fst,lst,stp;
   
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
         fin=fopen(log_file,"r");
         fdat=fopen(dat_file,"rb");

         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms1.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [ms1.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);
      }
   }
}


static void print_info(void)
{
   int isap,idfl;
   long ip;   
   lat_parms_t lat;
   
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
      if (noexp)
         printf("Configurations are read in imported file format\n\n");
      else
         printf("Configurations are read in exported file format\n\n");
         
      if (append)
      {
         printf("Random number generator:\n");
         printf("level = %d, seed = %d, effective seed = %d\n\n",
                level,seed,seed^(first-step));
      }
      else
      {
         printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
         printf("Open boundary conditions\n\n");

         printf("Random number generator:\n");
         printf("level = %d, seed = %d\n\n",level,seed);
         
         lat=lat_parms();
         printf("Lattice parameters:\n");
         printf("kappa_u = %.6f\n",lat.kappa_u);      
         printf("kappa_s = %.6f\n",lat.kappa_s);
         printf("kappa_c = %.6f\n",lat.kappa_c);               
         printf("csw = %.6f\n",lat.csw);      
         printf("cF = %.6f\n\n",lat.cF);

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


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;
   dfl_gen_parms_t dgp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();
   dgp=dfl_gen_parms();

   MAX(*nws,dp.Ns+2);
   MAX(*nwv,2*dpp.nkv+2);
   MAX(*nwv,2*dgp.nkv+2);
   MAX(*nwvd,4);
}


static void solver_wsize(int isp,int nsd,int np,
                         int *nws,int *nwsd,int *nwv,int *nwvd)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if (sp.solver==CGNE)
   {
      MAX(*nws,5);
      MAX(*nwsd,nsd+3);
   }
   else if (sp.solver==MSCG)
   {
      if (np>1)
      {
         MAX(*nwsd,nsd+np+3);
      }
      else
      {
         MAX(*nwsd,nsd+5);
      }
   }   
   else if (sp.solver==SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);
      MAX(*nwsd,nsd+2);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);      
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd);
   }         
}


static void reweight_wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nrw,irw,nsd;
   int n,*np,*isp,l;
   rw_parms_t rwp;
   solver_parms_t sp;

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;
   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);
      n=rwp.n;
      np=rwp.np;
      isp=rwp.isp;

      for (l=0;l<n;l++)
      {
         if ((rwp.rwfact==RWTM1)||(rwp.rwfact==RWTM1_EO)||
             (rwp.rwfact==RWTM2)||(rwp.rwfact==RWTM2_EO))
         {
            nsd=2;
            solver_wsize(isp[l],nsd,0,nws,nwsd,nwv,nwvd);
         }
         else if (rwp.rwfact==RWRAT)
         {
            sp=solver_parms(isp[l]);

            if (sp.solver==MSCG)
               nsd=3+np[l];
            else
               nsd=5;

            solver_wsize(isp[l],nsd,np[l],nws,nwsd,nwv,nwvd);
         }
         else
            error_root(1,1,"reweight_wsize [ms1.c]",
                       "Unknown reweighting factor");
      }
   }
}


static void alloc_rwstat(void)
{
   int nrw,irw,nmx,n;
   int **pp,*p;
   rw_parms_t rwp;

   nrw=file_head.nrw;
   nmx=0;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=rw_parms(irw);

      if (nmx<rwp.n)
         nmx=rwp.n;
   }

   nmx*=2;
   pp=malloc(nmx*sizeof(*pp));
   p=malloc(4*nmx*sizeof(*p));
   error((pp==NULL)||(p==NULL),1,"alloc_rwstat [ms1.c]",
         "Unable to allocate status array");

   rwstat=pp;   

   for (n=0;n<nmx;n++)
   {
      pp[n]=p;
      p+=4;
   }
}


static void print_rwstat(int irw)
{
   int n,*isp,nsrc;
   int nrs,l,j;
   rw_parms_t rwp;
   solver_parms_t sp;

   if (my_rank==0)
   {
      rwp=rw_parms(irw);
      n=rwp.n;
      isp=rwp.isp;
      nsrc=rwp.nsrc;
      nrs=0;

      for (l=0;l<n;l++)
      {
         for (j=0;j<3;j++)
            rwstat[l][j]=(rwstat[l][j]+(nsrc/2))/nsrc;

         nrs+=rwstat[l][3];
         sp=solver_parms(isp[l]);
         
         if (l==0)
            printf("RWF %d: status = ",irw);
         else
            printf(";");
         
         if (sp.solver==DFL_SAP_GCR)
            printf("%d,%d,%d",rwstat[l][0],rwstat[l][1],rwstat[l][2]);
         else
            printf("%d",rwstat[l][0]);
      }

      if (nrs)
         printf(" (no of subspace regenerations = %d)\n",nrs);
      else
         printf("\n");
   }
}


static void print_status(int irw,int *status)
{
   int nsrc,nrs,l;
   rw_parms_t rwp;
   solver_parms_t sp;   

   if (my_rank==0)
   {
      rwp=rw_parms(irw);
      nsrc=rwp.nsrc;

      for (l=0;l<3;l++)
      {
         status[l]=(status[l]+(nsrc/2))/nsrc;
         status[4+l]=(status[4+l]+(nsrc/2))/nsrc;
      }

      nrs=status[3]+status[7];
      sp=solver_parms(rwp.isp[0]);
      
      printf("RWF %d: status = ",irw);

      if (sp.solver==DFL_SAP_GCR)
         printf("%d,%d,%d",status[0],status[1],status[2]);
      else
         printf("%d",status[0]);

      if ((rwp.rwfact==RWTM2)||(rwp.rwfact==RWTM2_EO))
      {
         if (sp.solver==DFL_SAP_GCR)
            printf(";%d,%d,%d",status[4],status[5],status[6]);
         else
            printf(";%d",status[1]);
      }

      if (nrs)
         printf(" (no of subspace regenerations = %d)\n",nrs);
      else
         printf("\n");      
   }
}


static void set_data(int nc)
{
   int nrw,nsrc,irw,isrc;
   int n,l,j,status[8],stat[8];
   double *sqn,*lnr;
   rw_parms_t rwp;

   if (rwstat==NULL)
      alloc_rwstat();
   
   nrw=file_head.nrw;
   data.nc=nc;   

   for (irw=0;irw<nrw;irw++)
   {
      sqn=data.sqn[irw];
      lnr=data.lnr[irw];
      rwp=rw_parms(irw);
      nsrc=rwp.nsrc;
      set_sw_parms(sea_quark_mass(rwp.im0));
      
      if (rwp.rwfact==RWRAT)
      {
         n=rwp.n;

         for (l=0;l<(2*n);l++)
         {
            for (j=0;j<4;j++)
               rwstat[l][j]=0;
         }

         for (isrc=0;isrc<nsrc;isrc++)
         {
            lnr[isrc]=rwrat(rwp.irp,n,rwp.np,rwp.isp,sqn+isrc,rwstat+n);

            for (l=0;l<n;l++)
            {
               for (j=0;j<3;j++)
                  rwstat[l][j]+=rwstat[n+l][j];

               rwstat[l][3]+=(rwstat[n+l][3]!=0);
            }  
         }

         print_rwstat(irw);
      }
      else
      {
         for (l=0;l<8;l++)
         {
            status[l]=0;
            stat[l]=0;
         }

         for (isrc=0;isrc<nsrc;isrc++)
         {
            if (rwp.rwfact==RWTM1)
               lnr[isrc]=rwtm1(rwp.mu,rwp.isp[0],sqn+isrc,stat);
            else if (rwp.rwfact==RWTM1_EO)
               lnr[isrc]=rwtm1eo(rwp.mu,rwp.isp[0],sqn+isrc,stat);
            else if (rwp.rwfact==RWTM2)
               lnr[isrc]=rwtm2(rwp.mu,rwp.isp[0],sqn+isrc,stat);
            else if (rwp.rwfact==RWTM2_EO)
               lnr[isrc]=rwtm2eo(rwp.mu,rwp.isp[0],sqn+isrc,stat);
            else
               error_root(1,1,"set_data [ms1.c]","Unknown reweighting factor");

            for (l=0;l<3;l++)
            {
               status[l]+=stat[l];
               status[4+l]+=stat[4+l];
            }

            status[3]+=(stat[3]!=0);
            status[7]+=(stat[7]!=0);
         }

         print_status(irw,status);
      }

      if (my_rank==0)
      {
         printf("RWF %d: -ln(r) = %.4e",irw,lnr[0]);

         if (nsrc<=4)
         {
            for (isrc=1;isrc<nsrc;isrc++)
               printf(",%.4e",lnr[isrc]);
         }
         else
         {
            printf(",%.4e,...",lnr[1]);

            for (isrc=(nsrc-2);isrc<nsrc;isrc++)
               printf(",%.4e",lnr[isrc]);   
         }   

         printf("\n");
      }
   }
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
   int nws,nwsd,nwv,nwvd;
   double wt1,wt2,wtavg;
   dfl_parms_t dfl;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   alloc_data();
   check_files();
   print_info();
   dfl=dfl_parms();

   if (append)
      start_ranlux(level,seed^(first-step));
   else
      start_ranlux(level,seed);      
   geometry();

   reweight_wsize(&nws,&nwsd,&nwv,&nwvd);
   alloc_ws(nws);
   alloc_wsd(nwsd);
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   
   iend=0;   
   wtavg=0.0;
   
   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      
      if (my_rank==0)
         printf("Configuration no %d\n",nc);

      if (noexp)
      {
         save_ranlux();
         sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,nc,my_rank);
         read_cnfg(cnfg_file);
         restore_ranlux();
      }
      else
      {
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc);
         import_cnfg(cnfg_file);
      }

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

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);
      error_chk();

      if (my_rank==0)
      {
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));
         fflush(flog);         

         copy_file(log_file,log_save);
         copy_file(dat_file,dat_save);
      }

      check_endflag(&iend);
   }

   error_chk();
      
   if (my_rank==0)
   {
      fflush(flog);
      copy_file(log_file,log_save);
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
