
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Block-exporting and importing gauge, momentum and spinor fields.
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
#include "mdflds.h"
#include "sflds.h"
#include "su3fcts.h"
#include "linalg.h"
#include "archive.h"
#include "cmpflds.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)


int main(int argc,char *argv[])
{
   int my_rank,bc,nsize,n,eo,b,ip,ie;
   int nio_nodes,nio_streams,nb,ib,nl[4],bs[4];
   double phi[2],phi_prime[2],theta[3];
   su3_alg_dble *mom,**msv;
   su3_dble *udb,**usv;
   spinor_dble **wsd;
   mdflds_t *mdfs;
   char blk_dir[NAME_SIZE],blk_sub_dir[NAME_SIZE],cnfg[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   check_machine();

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Block-exporting and importing gauge, "
             "momentum and spinor fields\n");
      printf("-------------------------------------"
             "--------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("blk_dir","%s",blk_dir);
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("nio_nodes","%d",&nio_nodes);
      read_line("nio_streams","%d",&nio_streams);
      fclose(fin);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check3.c]",
                    "Syntax: check3 [-bc <type>]");
   }

   MPI_Bcast(blk_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_nodes,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nio_streams,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.5;
   theta[1]=1.0;
   theta[2]=-0.5;

   n=0;
   set_hmc_parms(1,&n,0,0,NULL,1,1.0);
   set_bc_parms(bc,1.0,1.0,1.0,1.0,phi,phi_prime,theta);
   print_bc_parms(3);

   start_ranlux(0,123456);
   geometry();
   alloc_wud(1);
   alloc_wfd(1);
   alloc_wsd(4);

   if (my_rank==0)
   {
      printf("Block-export random field configurations to disk. "
             "Then import the fields\n"
             "from there, compare with the saved fields and remove "
             "all files.\n\n");
   }

   usv=reserve_wud(1);
   msv=reserve_wfd(1);
   wsd=reserve_wsd(4);
   udb=udfld();
   mdfs=mdflds();
   mom=(*mdfs).mom;

   nl[0]=N0;
   nl[1]=N1;
   nl[2]=N2;
   nl[3]=N3;
   ib=blk_index(nl,bs,&nb);

   if (my_rank==0)
   {
      printf("Block size = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Number of blocks = %d\n",nb);
      printf("nio_nodes = %d\n",nio_nodes);
      printf("nio_streams = %d\n\n",nio_streams);
   }

   error_root((nb%nio_nodes)!=0,1,"main [check3.c]",
              "nb must be an integer multiple of nio_nodes");
   error((ib<0)||(ib>=nb),1,"main [check3.c]","Unexpected block index");
   n=ib/(nb/nio_nodes);
   b=ib%(nb/nio_nodes);
   set_nio_streams(nio_streams);

   nsize=name_size("%s/%d/%d/test.ud_b%d",
                   blk_dir,nio_nodes-1,(nb/nio_nodes)-1,nb-1);
   error_root(nsize>=NAME_SIZE,1,"main [check3.c]",
              "blk_dir name is too long");
   sprintf(blk_sub_dir,"%s/%d/%d",blk_dir,n,b);

   ip=((((cpr[0]*L0)%bs[0])==0)&&(((cpr[1]*L1)%bs[1])==0)&&
       (((cpr[2]*L2)%bs[2])==0)&&(((cpr[3]*L3)%bs[3])==0));

   if (ip)
      check_dir(blk_sub_dir);

   random_ud();
   random_mom();
   random_sd(VOLUME,wsd[0],1.0);
   random_sd(VOLUME,wsd[1],1.0);

   cm3x3_assign(4*VOLUME,udb,usv[0]);
   assign_alg2alg(4*VOLUME,mom,msv[0]);
   assign_sd2sd(VOLUME,wsd[0],wsd[2]);
   assign_sd2sd(VOLUME,wsd[1],wsd[3]);

   print_flags();

   sprintf(cnfg,"%s/test.ud_b%d",blk_sub_dir,ib);
   blk_export_cnfg(bs,cnfg);
   sprintf(cnfg,"%s/test.fd_b%d",blk_sub_dir,ib);
   blk_export_mfld(bs,cnfg);

   for (eo=0;eo<2;eo++)
   {
      sprintf(cnfg,"%s/test.sd%d_b%d",blk_sub_dir,eo,ib);
      blk_export_sfld(bs,cnfg,eo,wsd[eo]);
   }

   random_ud();
   random_mom();
   random_sd(VOLUME,wsd[0],1.0);
   random_sd(VOLUME/2,wsd[1],1.0);

   sprintf(cnfg,"%s/test.ud_b%d",blk_sub_dir,ib);
   blk_import_cnfg(cnfg,0x0);

   sprintf(cnfg,"%s/test.fd_b%d",blk_sub_dir,ib);
   blk_import_mfld(cnfg);
   if (ip)
      remove(cnfg);

   for (eo=0;eo<2;eo++)
   {
      sprintf(cnfg,"%s/test.sd%d_b%d",blk_sub_dir,eo,ib);
      blk_import_sfld(cnfg,eo,wsd[eo]);
      if (ip)
         remove(cnfg);
   }

   ie=(check_bc(0.0)^0x1);
   ie|=check_ud(4*VOLUME,udb,usv[0]);
   error(ie!=0,1,"main [check3.c]",
         "The gauge field is not properly restored");

   ie=check_fd(4*VOLUME,mom,msv[0]);
   error(ie!=0,1,"main [check3.c]",
         "The momentum field is not properly restored");

   for (eo=0;eo<2;eo++)
   {
      ie=check_sd(VOLUME,wsd[eo],wsd[eo+2]);
      error(ie!=0,1,"main [check3.c]",
            "The spinor field is not properly restored (eo=%d)",eo);
   }

   print_flags();

   if (my_rank==0)
   {
      printf("No errors detected --- the fields are correctly exported\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
