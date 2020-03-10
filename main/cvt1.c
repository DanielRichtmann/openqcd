
/*******************************************************************************
*
* File cvt1.c
*
* Copyright (C) 2017-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Conversion of gauge-field configurations from one storage format to another.
*
* Syntax: cvt1 -i <filename> [-rmold]
*
* For usage instructions see the file README.cvt1.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
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

static int my_rank,rmold;
static int first,last,step;
static char line[NAME_SIZE];
static char nbase[NAME_SIZE],log_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fend=NULL;


static void read_dirs(void)
{
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Log directory");
      read_line("log_dir","%s",log_dir);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   check_dir_root(log_dir);

   error(name_size("%s/%s.log~",log_dir,nbase)>=NAME_SIZE,1,
         "read_dirs [cvt1.c]","log_dir name is too long");
   sprintf(log_file,"%s/%s.log",log_dir,nbase);
   sprintf(end_file,"%s/%s.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
}


static void read_bc_parms(void)
{
   int bc;
   double phi[2],phi_prime[2],theta[3];

   if (my_rank==0)
   {
      find_section("Boundary conditions");
      read_line("type","%d",&bc);

      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;

      if (bc==1)
         read_dprms("phi",2,phi);

      if ((bc==1)||(bc==2))
         read_dprms("phi'",2,phi_prime);
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);

   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;

   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
}


static void read_range(void)
{
   if (my_rank==0)
   {
      find_section("Configurations");
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);

      error_root((first<1)||(last<first)||(step<1)||((last-first)%step!=0),1,
                 "read_range [cvt1.c]","Improper configuration range");
   }

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void read_iodat(void)
{
   int i,type,nion,nios,bs[4];

   for (i=0;i<2;i++)
   {
      if (my_rank==0)
      {
         if (i==0)
            find_section("Input storage format");
         else
            find_section("Output storage format");

         read_line("type","%s",line);

         if (strchr(line,'e')!=NULL)
            type=0x1;
         else if (strchr(line,'b')!=NULL)
            type=0x2;
         else if (strchr(line,'l')!=NULL)
            type=0x4;
         else
            type=0x0;

         error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [cvt1.c]",
                    "Improper configuration storage type");

         read_line("cnfg_dir","%s",line);

         if ((i==1)&&(type&0x2))
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

      iodat[i].type=type;
      strcpy(iodat[i].cnfg_dir,line);
      iodat[i].nio_nodes=nion;
      iodat[i].nio_streams=nios;
      iodat[i].bs[0]=bs[0];
      iodat[i].bs[1]=bs[1];
      iodat[i].bs[2]=bs[2];
      iodat[i].bs[3]=bs[3];
   }
}


static void read_infile(int argc,char *argv[])
{
   int ifile;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      rmold=find_opt(argc,argv,"-rmold");

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [cvt1.c]",
                 "Syntax: cvt1 -i <filename> [-rmold]");

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [cvt1.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&rmold,1,MPI_INT,0,MPI_COMM_WORLD);

   read_dirs();
   read_bc_parms();
   read_range();
   read_iodat();

   if (my_rank==0)
      fclose(fin);
}


static void check_files(void)
{
   int i,ns[4],bs[4];
   int type,nion,nb,ib,n;
   char *cnfg_dir;

   for (i=0;i<2;i++)
   {
      type=iodat[i].type;
      cnfg_dir=iodat[i].cnfg_dir;
      iodat[i].nb=NPROC;
      iodat[i].ib=0;

      if (type&0x1)
      {
         error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
               "check_files [cvt1.c]","cnfg_dir name is too long");
         check_dir_root(cnfg_dir);

         if (i==0)
         {
            sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
            lat_sizes(line,ns);
            error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                       "check_files [cvt1.c]","Lattice size mismatch");
         }
      }
      else if (type&0x2)
      {
         if (i==0)
         {
            error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,
                  1,"check_files [cvt1.c]","cnfg_dir name is too long");
            sprintf(line,"%s/0/0",cnfg_dir);
            if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
               check_dir(line);

            sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
            blk_sizes(line,ns,bs);
            error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                       "check_files [cvt1.c]","Lattice size mismatch");
            iodat[i].bs[0]=bs[0];
            iodat[i].bs[1]=bs[1];
            iodat[i].bs[2]=bs[2];
            iodat[i].bs[3]=bs[3];
         }
         else
         {
            ns[0]=N0;
            ns[1]=N1;
            ns[2]=N2;
            ns[3]=N3;
            bs[0]=iodat[i].bs[0];
            bs[1]=iodat[i].bs[1];
            bs[2]=iodat[i].bs[2];
            bs[3]=iodat[i].bs[3];
         }

         ib=blk_index(ns,bs,&nb);
         nion=iodat[i].nio_nodes;
         n=nb/nion;
         error_root(nb%nion!=0,1,"check_files [cvt1.c]",
                    "Number of blocks is not a multiple of nio_nodes");
         error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,
                         nbase,last,nb-1)>=NAME_SIZE,1,
               "check_files [cvt1.c]","cnfg_dir name is too long");
         sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
         strcpy(cnfg_dir,line);
         if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
             ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0)&&(ib<nb))
            check_dir(cnfg_dir);

         iodat[i].nb=nb;
         iodat[i].ib=ib;
      }
      else
      {
         nion=iodat[i].nio_nodes;
         n=NPROC/nion;
         error_root(NPROC%nion!=0,1,"check_files [cvt1.c]",
                    "Number of processes is not a multiple of nio_nodes");
         error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,nbase,
                         last,NPROC-1)>=NAME_SIZE,1,
               "check_files [cvt1.c]","cnfg_dir name is too long");
         sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
         strcpy(cnfg_dir,line);
         check_dir(cnfg_dir);
      }
   }
}


static void read_ud(int icnfg)
{
   int type,nios,ib;
   char *cnfg_dir;

   type=iodat[0].type;
   cnfg_dir=iodat[0].cnfg_dir;
   nios=iodat[0].nio_streams;
   ib=iodat[0].ib;

   if (type&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file,0x0);
   }
   else if (type&0x2)
   {
      set_nio_streams(nios);
      sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,ib);
      blk_import_cnfg(cnfg_file,0x0);
   }
   else
   {
      set_nio_streams(nios);
      sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
      read_cnfg(cnfg_file);
   }
}


static void save_ud(int icnfg)
{
   int type,nios,ib,*bs;
   char *cnfg_dir;

   type=iodat[1].type;
   cnfg_dir=iodat[1].cnfg_dir;
   nios=iodat[1].nio_streams;
   ib=iodat[1].ib;
   bs=iodat[1].bs;

   if (type&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      export_cnfg(cnfg_file);
   }
   else if (type&0x2)
   {
      set_nio_streams(nios);
      sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,ib);
      blk_export_cnfg(bs,cnfg_file);
   }
   else
   {
      set_nio_streams(nios);
      sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
      write_cnfg(cnfg_file);
   }
}


static void print_info(void)
{
   int i,type;
   long ip;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      flog=freopen(log_file,"w",stdout);
      error_root(flog==NULL,1,"print_info [cvt1.c]","Unable to open log file");

      printf("\n");
      printf("Conversion of gauge-field configurations\n");
      printf("----------------------------------------\n\n");

      printf("Program version %s\n",openQCD_RELEASE);

      printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
      printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d process block size\n\n",
             NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);

      print_bc_parms(0);

      for (i=0;i<2;i++)
      {
         if (i==0)
            printf("Input configuration storage type = ");
         else
            printf("Output configuration storage type = ");

         type=iodat[i].type;

         if (type&0x1)
            printf("exported\n");
         else if (type&0x2)
            printf("block-exported\n");
         else
            printf("local\n");

         if (type&0x6)
         {
            printf("Parallel I/O parameters: "
                   "nio_nodes = %d, nio_streams = %d\n",
                   iodat[i].nio_nodes,iodat[i].nio_streams);
         }

         if (type&0x2)
         {
            printf("Block size = %dx%dx%dx%d\n",iodat[i].bs[0],
                   iodat[i].bs[1],iodat[i].bs[2],iodat[i].bs[3]);
         }

         printf("\n");
      }

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);

      if (rmold)
         printf("Old configurations are deleted\n\n");

      fflush(flog);
   }
}


static void print_log(int icnfg,double *wt,double *wtall)
{
   double r;

   if (my_rank==0)
   {
      r=(double)((icnfg-first)/step+1);

      printf("Configuration no %d fully processed in %.2e sec "
             "(average = %.2e)\n",icnfg,wt[0]+wt[1],(wtall[0]+wtall[1])/r);
      printf("Average read time = %.2e, average write time = %.2e\n\n",
             wtall[0]/r,wtall[1]/r);

      fflush(flog);
   }
}


static void remove_cnfg(int icnfg)
{
   int type,nios,n,i;
   int nb,ib,ip,*bs;
   char *cnfg_dir;

   if (rmold)
   {
      MPI_Barrier(MPI_COMM_WORLD);

      type=iodat[0].type;
      cnfg_dir=iodat[0].cnfg_dir;

      if (type&0x1)
      {
         if (my_rank==0)
         {
            sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
            remove(cnfg_file);
         }
      }
      else if (type&0x2)
      {
         nios=iodat[0].nio_streams;
         nb=iodat[0].nb;
         ib=iodat[0].ib;
         bs=iodat[0].bs;
         ip=(((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
             ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0));
         n=nb/nios;

         for (i=0;i<n;i++)
         {
            if ((i==(ib%n))&&(ip!=0))
            {
               sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,ib);
               remove(cnfg_file);
            }

            MPI_Barrier(MPI_COMM_WORLD);
         }
      }
      else
      {
         nios=iodat[0].nio_streams;
         n=NPROC/nios;

         for (i=0;i<n;i++)
         {
            if (i==(my_rank%n))
            {
               sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
               remove(cnfg_file);
            }

            MPI_Barrier(MPI_COMM_WORLD);
         }
      }
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
   int icnfg,iend;
   double wt1,wt2,wt3,wt[2],wtall[2];

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   geometry();
   check_files();
   print_info();

   iend=0;
   wtall[0]=0.0;
   wtall[1]=0.0;

   for (icnfg=first;(iend==0)&&(icnfg<=last);icnfg+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      read_ud(icnfg);

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      save_ud(icnfg);

      MPI_Barrier(MPI_COMM_WORLD);
      wt3=MPI_Wtime();

      wt[0]=wt2-wt1;
      wt[1]=wt3-wt2;
      wtall[0]+=wt[0];
      wtall[1]+=wt[1];
      print_log(icnfg,wt,wtall);

      if (my_rank==0)
      {
         fflush(flog);
         copy_file(log_file,log_save);
      }

      remove_cnfg(icnfg);
      check_endflag(&iend);
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
