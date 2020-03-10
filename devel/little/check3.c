
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2007-2016, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Direct check of Aw_dble() and Aw().
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
#include "vflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "dfl.h"
#include "little.h"
#include "global.h"


static void random_basis(int Ns)
{
   int i;
   spinor **ws;

   ws=reserve_ws(Ns);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }

   dfl_subspace(ws);
   release_ws();
}


int main(int argc,char *argv[])
{
   int my_rank,bc;
   int bs[4],Ns,nb,nv;
   int ieo,im0,imu;
   double phi[2],phi_prime[2],theta[3];
   double m0[2],mu[2],dev;
   qflt rqsm;
   complex **wv,z;
   complex_dble **wvd,zd;
   spinor **ws;
   FILE *fin=NULL,*flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Direct check of Aw_dble() and Aw()\n");
      printf("----------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>]");
   }

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,0,1.978);
   print_lat_parms();

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc,1.0,1.0,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(2);

   set_dfl_parms(bs,Ns);
   start_ranlux(0,123456);
   geometry();

   alloc_ws(Ns+2);
   alloc_wv(3);
   alloc_wvd(2);

   ws=reserve_ws(2);
   wv=reserve_wv(3);
   wvd=reserve_wvd(2);
   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;

   random_ud();
   set_ud_phase();
   random_basis(Ns);

   m0[0]=-0.0123;
   m0[1]= 0.0257;

   mu[0]=0.0157;
   mu[1]=0.0239;

   for (ieo=0;ieo<2;ieo++)
   {
      set_tm_parms(ieo);

      for (im0=0;im0<2;im0++)
      {
         set_sw_parms(m0[im0]);

         for (imu=0;imu<2;imu++)
         {
            set_Aw(mu[imu]);

            random_v(nv,wv[0],1.0f);
            assign_v2vd(nv,wv[0],wvd[0]);
            Aw_dble(wvd[0],wvd[1]);
            Aw(wv[0],wv[1]);
            assign_v2vd(nv,wv[1],wvd[0]);

            zd.re=-1.0;
            zd.im=0.0;
            mulc_vadd_dble(nv,wvd[0],wvd[1],zd);
            rqsm=vnorm_square_dble(nv,1,wvd[0]);
            dev=rqsm.q[0];
            rqsm=vnorm_square_dble(nv,1,wvd[1]);
            dev/=rqsm.q[0];

            if (my_rank==0)
               printf("Relative deviations (ieo=%d,im0=%d,imu=%d): "
                      "Aw_dble vs Aw = %.1e",ieo,im0,imu,sqrt(dev));

            random_v(nv,wv[0],1.0f);
            Aw(wv[0],wv[1]);

            sw_term(NO_PTS);
            assign_ud2u();
            assign_swd2sw();
            dfl_v2s(wv[0],ws[0]);
            Dw((float)(mu[imu]),ws[0],ws[1]);
            dfl_s2v(ws[1],wv[2]);

            z.re=-1.0f;
            z.im=0.0f;
            mulc_vadd(nv,wv[2],wv[1],z);
            dev=(double)(vnorm_square(nv,1,wv[2])/vnorm_square(nv,1,wv[1]));

            if (my_rank==0)
               printf(", Aw vs Dw = %.1e\n",sqrt(dev));
         }
      }
   }

   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
