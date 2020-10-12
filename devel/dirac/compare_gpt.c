
/*******************************************************************************
*
* File compare_gpt.c
*
* Copyright (C) 2005, 2008, 2011-2013, 2016, 2018-2020 Martin Luescher, Daniel Richtmann
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Apply Dw_dble() and write fields to files. Used for comparison with gpt.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"
#include "archive.h"


int main(int argc,char *argv[])
{
   int my_rank,bc;
   int nflds;
   double phi[2],phi_prime[2],theta[3];
   double mu,beta,c0,csw,kappa[1];
   double cG, cG_prime, cF, cF_prime;
   spinor_dble **psd;
   FILE *flog=NULL;
   int i;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("compare_gpt.log","w",stdout);

      printf("\n");
      printf("Apply Dw_dble() and write fields to files. Used for comparison with gpt.\n");
      printf("------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      if (NPROC>1)
         printf("There are %d MPI processes\n",NPROC);
      else
         printf("There is 1 MPI process\n");

      if ((VOLUME*sizeof(double))<(64*1024))
      {
         printf("The local size of the gauge field is %d KB\n",
                (int)((72*VOLUME*sizeof(double))/(1024)));
         printf("The local size of a quark field is %d KB\n",
                (int)((24*VOLUME*sizeof(double))/(1024)));
      }
      else
      {
         printf("The local size of the gauge field is %d MB\n",
                (int)((72*VOLUME*sizeof(double))/(1024*1024)));
         printf("The local size of a quark field is %d MB\n",
                (int)((24*VOLUME*sizeof(double))/(1024*1024)));
      }

#if (defined x64)
#if (defined AVX)
#if (defined FMA3)
   printf("Using AVX and FMA3 instructions\n");
#else
   printf("Using AVX instructions\n");
#endif
#else
      printf("Using SSE3 instructions and 16 xmm registers\n");
#endif
#if (defined P3)
      printf("Assuming SSE prefetch instructions fetch 32 bytes\n");
#elif (defined PM)
      printf("Assuming SSE prefetch instructions fetch 64 bytes\n");
#elif (defined P4)
      printf("Assuming SSE prefetch instructions fetch 128 bytes\n");
#else
      printf("SSE prefetch instructions are not used\n");
#endif
#endif
      printf("\n");

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [compare_gpt.c]",
                    "Syntax: compare_gpt [-bc <type>]");
   }

   c0 = 1.0;
   cF = 1.3;
   cF_prime = 1.3;
   cG = 1.0;
   cG_prime = 1.0;
   beta = 5.5;
   csw = 1.978;
   kappa[0] = 0.13500;
   set_lat_parms(beta,c0,1,kappa,0,csw);
   print_lat_parms();

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.000;
   phi[1]=0.000;
   phi_prime[0]=0.000;
   phi_prime[1]=0.000;
   theta[0]=0.000;
   theta[1]=0.000;
   theta[2]=0.000;
   set_bc_parms(bc,cG,cG_prime,cF,cF_prime,phi,phi_prime,theta);
   print_bc_parms(3);

   geometry();

   set_sw_parms(1.0/(2.0*kappa[0])-4.0);
   mu=0.0;

   random_ud();
   export_cnfg("field_gauge.bin");

   set_ud_phase();
   sw_term(NO_PTS);

   nflds=2;
   alloc_wsd(nflds);
   psd=reserve_wsd(nflds);

   for (i=0;i<nflds;i++)
      random_sd(VOLUME,psd[i],1.0);

   Dw_dble(mu,psd[0],psd[1]);

   export_sfld("field_src.bin",0,psd[0]);
   export_sfld("field_dst.bin",0,psd[1]);

   fclose(flog);

   MPI_Finalize();
   exit(0);
}
