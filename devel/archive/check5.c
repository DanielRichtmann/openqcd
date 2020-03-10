
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Block-importing the gauge field previously exported by check3. This program
* accepts the options
*
*  -bc <int>        Sets the boundary conditions (0: open, 1: SF,
*                   2: open-SF, 3: periodic).
*
*  -mask <int>      Extension mask (4 bit unsigned integer).
*
*  -rmold           On exit the gauge-field configuration file is
*                   deleted if this option is set.
*
* None of these options need to be set. The default bc and mask values are 0
* and configurations are not deleted.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "su3fcts.h"
#include "tcharge.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int my_rank,bc,mask,rmold;
static char blk_dir[NAME_SIZE],blk_sub_dir[NAME_SIZE],cnfg_file[NAME_SIZE];


static double get_plaq(char *in)
{
   int mu,ip,ir,endian,nl[4];
   stdint_t lsize[12];
   double plaq;
   FILE *fin;

   endian=endianness();

   for (mu=0;mu<4;mu++)
      nl[mu]=0;
   ip=ipr_global(nl);

   if (my_rank==ip)
   {
      fin=fopen(in,"rb");
      error_loc(fin==NULL,1,"get_plaq [check5.c]",
                "Unable to open input file");
      ir=fread(lsize,sizeof(stdint_t),12,fin);
      ir+=fread(&plaq,sizeof(double),1,fin);
      error_loc(ir!=13,1,"get_plaq [check5.c]",
                "Incorrect read count");
      fclose(fin);

      if (endian==BIG_ENDIAN)
         bswap_double(1,&plaq);
   }

   MPI_Bcast(&plaq,1,MPI_DOUBLE,ip,MPI_COMM_WORLD);

   return plaq;
}


int main(int argc,char *argv[])
{
   int n,ie,nb,ib,nl[4],bs[4];
   int nio_nodes,nio_streams;
   double phi[2],phi_prime[2],theta[3];
   double plaq0,plaq1,Q;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   check_machine();

   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Block-importing a previously exported gauge field\n");
      printf("-------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("blk_dir","%s",blk_dir);
      read_line("nio_nodes","%d",&nio_nodes);
      read_line("nio_streams","%d",&nio_streams);
      fclose(fin);

      ie=1;
      n=find_opt(argc,argv,"-bc");
      if (n>0)
         ie&=sscanf(argv[n+1],"%d",&bc);
      else
         bc=0;

      n=find_opt(argc,argv,"-mask");
      if (n>0)
         ie&=sscanf(argv[n+1],"%i",&mask);
      else
         mask=0x0;

      n=find_opt(argc,argv,"-rmold");
      if (n>0)
         rmold=1;
      else
         rmold=0;

      error_root(ie!=1,1,"main [check5.c]",
                 "Syntax check5 [-bc <int>] [-mask <int>] [-rmold]");
      error_root(name_size("%s/test.ud",blk_dir)>=NAME_SIZE,1,
                 "main [check5.c]","blk_dir name is too long");
      error_root((bc<0)||(bc>3)||(mask<0x0)||(mask>0xf),1,
                 "main [check5.c]","Argument bc or mask is out of range");
   }

   MPI_Bcast(blk_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_nodes,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_streams,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&mask,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rmold,1,MPI_INT,0,MPI_COMM_WORLD);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;

   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(1);

   geometry();
   random_ud();

   error(name_size("%s/0/0/test.ud_b0",blk_dir)>=NAME_SIZE,1,
         "main [check5.c]","blk_dir name is too long");
   sprintf(blk_sub_dir,"%s/0/0",blk_dir);
   if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
      check_dir(blk_sub_dir);

   sprintf(cnfg_file,"%s/test.ud_b0",blk_sub_dir);
   blk_sizes(cnfg_file,nl,bs);
   plaq0=get_plaq(cnfg_file);

   if (my_rank==0)
   {
      printf("Configuration file = %s/k/i/test.ud_bj\n",blk_dir);
      printf("Extension mask = %#x\n",(unsigned int)(mask));

      printf("Previous lattice size = %dx%dx%dx%d\n",
             nl[0],nl[1],nl[2],nl[3]);
      printf("Previous block size = %dx%dx%dx%d\n",
             bs[0],bs[1],bs[2],bs[3]);
      printf("Previous average plaquette = %.16e\n\n",plaq0);
   }

   ib=blk_index(nl,bs,&nb);
   n=nb/nio_nodes;
   error_root(nb%nio_nodes!=0,1,"check_files [check5.c]",
              "Number of blocks is not a multiple of nio_nodes");
   error(name_size("%s/%d/%d/test.ud_b%d",blk_dir,nio_nodes-1,n-1,nb-1)
         >=NAME_SIZE,1,"main [check5.c]","blk_dir name is too long");
   sprintf(blk_sub_dir,"%s/%d/%d",blk_dir,ib/n,ib%n);
   if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
       ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0)&&(ib<nb))
      check_dir(blk_sub_dir);

   set_nio_streams(nio_streams);
   sprintf(cnfg_file,"%s/test.ud_b%d",blk_sub_dir,ib);
   blk_import_cnfg(cnfg_file,mask);
   plaq1=plaq_sum_dble(1)/((double)(6*N0*N1)*(double)(N2*N3));
   Q=tcharge();

   if (my_rank==0)
   {
      printf("Block-imported configuration:\n");
      printf("Current average plaquette  = %.16e\n",plaq1);
      printf("Relative difference = %.1e (DBL_EPSILON = %.1e)\n",
             fabs(1.0-plaq0/plaq1),DBL_EPSILON);
      printf("Topological charge = %.4e\n\n",Q);
   }

   if ((my_rank==0)&&(rmold))
      remove(cnfg_file);

   if (my_rank==0)
   {
      if (rmold)
         printf("Field configuration deleted\n\n");
      else
         printf("Field configuration preserved\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
