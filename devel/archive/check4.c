
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Importing the gauge field previously exported by check2. This program
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
#include "tcharge.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int my_rank,bc,mask,rmold;
static char cnfg_dir[NAME_SIZE],cnfg_file[NAME_SIZE];


static double get_plaq(char *in)
{
   int ir,endian;
   stdint_t lsize[4];
   double plaq;
   FILE *fin=NULL;

   endian=endianness();

   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"get_plaq [check4.c]",
                 "Unable to open input file");
      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&plaq,sizeof(double),1,fin);
      error_root(ir!=5,1,"get_plaq [check4.c]",
                 "Incorrect read count");
      fclose(fin);

      if (endian==BIG_ENDIAN)
         bswap_double(1,&plaq);
   }

   MPI_Bcast(&plaq,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   return plaq;
}


int main(int argc,char *argv[])
{
   int n,ie,nl[4];
   double phi[2],phi_prime[2],theta[3];
   double plaq0,plaq1,Q;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   check_machine();

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      fin=freopen("check2.in","r",stdin);

      printf("\n");
      printf("Importing a previously exported gauge field\n");
      printf("-------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("cnfg_dir","%s",cnfg_dir);
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

      error_root(ie!=1,1,"main [check4.c]",
                 "Syntax check4 [-bc <int>] [-mask <int>] [-rmold]");
      error_root((bc<0)||(bc>3)||(mask<0x0)||(mask>0xf),1,
                 "main [check4.c]","Argument bc or mask is out of range");
   }

   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
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

   error_root(name_size("%s/test.ud",cnfg_dir)>=NAME_SIZE,1,
              "main [check4.c]","cnfg_dir name is too long");
   sprintf(cnfg_file,"%s/test.ud",cnfg_dir);
   check_dir_root(cnfg_dir);
   lat_sizes(cnfg_file,nl);
   plaq0=get_plaq(cnfg_file);

   if (my_rank==0)
   {
      printf("Configuration file = %s\n",cnfg_file);
      printf("Extension mask = %#x\n",(unsigned int)(mask));

      printf("Previous lattice size = %dx%dx%dx%d\n",
             nl[0],nl[1],nl[2],nl[3]);
      printf("Previous average plaquette = %.16e\n\n",plaq0);
   }

   import_cnfg(cnfg_file,mask);
   plaq1=plaq_sum_dble(1)/((double)(6*N0*N1)*(double)(N2*N3));
   Q=tcharge();

   if (my_rank==0)
   {
      printf("Imported configuration:\n");
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
