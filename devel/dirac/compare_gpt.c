
/*******************************************************************************
*
* File compare_gpt.c
*
* Copyright (C) 2005, 2008, 2011-2013, 2016, 2018-2020 Martin Luescher, Daniel Richtmann
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Apply Dirac operator methods and write fields to files. Used for comparison with gpt.
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
   int nflds,nsrcs;
   double phi[2],phi_prime[2],theta[3];
   double mu,beta,c0,csw,kappa[1];
   double cG, cG_prime, cF, cF_prime;
   spinor_dble **psd;
   FILE *flog=NULL;
   int i;
   char fname[NAME_SIZE];

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("compare_gpt.log","w",stdout);

      printf("\n");
      printf("Apply Dirac operator methods and write fields to files. Used for comparison with gpt.\n");
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
   sprintf(fname,"gauge.bc_%d.bin",bc); export_cnfg(fname);

   set_ud_phase();
   sw_term(NO_PTS);

   nflds=13; /* 6 source vectors and dst vectors, 1 temporary */
   nsrcs=nflds/2;
   alloc_wsd(nflds);
   psd=reserve_wsd(nflds);

   /* set all fields to random values */
   for (i=0;i<nflds;i++)
      random_sd(NSPIN,psd[i],1.0);

   /* use same source for all kernels */
   for (i=1;i<nsrcs;i++)
      assign_sd2sd(NSPIN,psd[0],psd[i]);

   /* set destination fields to zero */
   for (i=nsrcs;i<nflds;i++)
      set_sd2zero(NSPIN,psd[i]);

   /* apply Dirac operator methods */
   Dw_dble(mu,psd[0],psd[6]);
   Dwoo_dble(mu,psd[1],psd[7]);
   Dwee_dble(mu,psd[2],psd[8]);
   Dwoe_dble(psd[3],psd[9]);

   /* this is special since it does r=r_in-Dweo*s -> extract Dweo*s = r_in-r */
   assign_sd2sd(NSPIN,psd[10],psd[12]); /* use last as temporary */
   Dweo_dble(psd[4],psd[10]);
   mulr_spinor_add_dble(VOLUME/2,psd[12],psd[10],-1.0);
   assign_sd2sd(NSPIN,psd[12],psd[10]);

   /* use Mdag = g5*M*g5 */
   assign_sd2sd(NSPIN,psd[5],psd[12]);
   mulg5_dble(VOLUME,psd[12]);
   Dw_dble(mu,psd[12],psd[11]);
   mulg5_dble(VOLUME,psd[11]);

   sprintf(fname,"src.bc_%d.bin",bc); export_sfld(fname,0,psd[0]);
   sprintf(fname,"dst_Dw_dble.bc_%d.bin",bc); export_sfld(fname,0,psd[6]);
   sprintf(fname,"dst_Dwoo_dble.bc_%d.bin",bc); export_sfld(fname,0,psd[7]);
   sprintf(fname,"dst_Dwee_dble.bc_%d.bin",bc); export_sfld(fname,0,psd[8]);
   sprintf(fname,"dst_Dwoe_dble.bc_%d.bin",bc); export_sfld(fname,0,psd[9]);
   sprintf(fname,"dst_Dweo_dble.bc_%d.bin",bc); export_sfld(fname,0,psd[10]);
   sprintf(fname,"dst_Dwdag_dble.bc_%d.bin",bc); export_sfld(fname,0,psd[11]);

   fclose(flog);

   MPI_Finalize();
   exit(0);
}
