
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005-2018 Martin Luescher, Filippo Palombi,
*                         Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Conservation of the Hamilton function by the MD evolution.
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
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "archive.h"
#include "forces.h"
#include "dfl.h"
#include "update.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static struct
{
   int type,nio_nodes,nio_streams;
   int nb,ib;
   char cnfg_dir[NAME_SIZE];
} iodat;

static int my_rank,first,last,step;
static char line[NAME_SIZE],nbase[NAME_SIZE];
static char cnfg_file[NAME_SIZE];
static FILE *flog=NULL,*fin=NULL;


static void read_iodat(void)
{
   int type,nion,nios;

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

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

      error_root((strlen(line)!=1)||(type==0x0),1,"read_iodat [check3.c]",
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
                 "read_iodat [check3.c]","Improper configuration range");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(line,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat.type=type;
   strcpy(iodat.cnfg_dir,line);
   iodat.nio_nodes=nion;
   iodat.nio_streams=nios;
}


static void read_lat_parms(void)
{
   int nk,isw;
   double beta,c0,csw,*kappa;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("beta","%lf",&beta);
      read_line("c0","%lf",&c0);
      nk=count_tokens("kappa");
      read_line("isw","%d",&isw);
      read_line("csw","%lf",&csw);
   }

   MPI_Bcast(&beta,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&c0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nk,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&isw,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   if (nk>0)
   {
      kappa=malloc(nk*sizeof(*kappa));
      error(kappa==NULL,1,"read_lat_parms [check3.c]",
            "Unable to allocate parameter array");
      if (my_rank==0)
         read_dprms("kappa",nk,kappa);
      MPI_Bcast(kappa,nk,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      kappa=NULL;

   set_lat_parms(beta,c0,nk,kappa,isw,csw);

   if (nk>0)
      free(kappa);
}


static void read_bc_parms(void)
{
   int bc;
   double cG,cG_prime,cF,cF_prime;
   double phi[2],phi_prime[2],theta[3];

   find_section("Boundary conditions");
   read_line("type","%d",&bc);

   phi[0]=0.0;
   phi[1]=0.0;
   phi_prime[0]=0.0;
   phi_prime[1]=0.0;
   cG=1.0;
   cG_prime=1.0;
   cF=1.0;
   cF_prime=1.0;

   if (bc==1)
      read_dprms("phi",2,phi);

   if ((bc==1)||(bc==2))
      read_dprms("phi'",2,phi_prime);

   if (bc!=3)
   {
      read_line("cG","%lf",&cG);
      read_line("cF","%lf",&cF);
   }

   if (bc==2)
   {
      read_line("cG'","%lf",&cG_prime);
      read_line("cF'","%lf",&cF_prime);
   }

   read_dprms("theta",3,theta);

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(theta,3,MPI_DOUBLE,0,MPI_COMM_WORLD);

   set_bc_parms(bc,cG,cG_prime,cF,cF_prime,phi,phi_prime,theta);
}


static void read_hmc_parms(void)
{
   int nact,*iact;
   int npf,nmu,nlv;
   double tau,*mu;

   if (my_rank==0)
   {
      find_section("HMC parameters");
      nact=count_tokens("actions");
      read_line("npf","%d",&npf);
      nmu=count_tokens("mu");
      read_line("nlv","%d",&nlv);
      read_line("tau","%lf",&tau);
   }

   MPI_Bcast(&nact,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&npf,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmu,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nlv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&tau,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   if (nact>0)
   {
      iact=malloc(nact*sizeof(*iact));
      error(iact==NULL,1,"read_hmc_parms [check3.c]",
            "Unable to allocate temporary array");
      if (my_rank==0)
         read_iprms("actions",nact,iact);
      MPI_Bcast(iact,nact,MPI_INT,0,MPI_COMM_WORLD);
   }
   else
      iact=NULL;

   if (nmu>0)
   {
      mu=malloc(nmu*sizeof(*mu));
      error(mu==NULL,1,"read_hmc_parms [check3.c]",
            "Unable to allocate temporary array");
      if (my_rank==0)
         read_dprms("mu",nmu,mu);
      MPI_Bcast(mu,nmu,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      mu=NULL;

   set_hmc_parms(nact,iact,npf,nmu,mu,nlv,tau);

   if (nact>0)
      free(iact);
   if (nmu>0)
      free(mu);
}


static void read_integrator(void)
{
   int nlv,i,j,k,l;
   hmc_parms_t hmc;
   mdint_parms_t mdp;
   force_parms_t fp;
   rat_parms_t rp;

   hmc=hmc_parms();
   nlv=hmc.nlv;

   for (i=0;i<nlv;i++)
   {
      read_mdint_parms(i);
      mdp=mdint_parms(i);

      for (j=0;j<mdp.nfr;j++)
      {
         k=mdp.ifr[j];
         fp=force_parms(k);

         if (fp.force==FORCES)
            read_force_parms2(k);

         fp=force_parms(k);

         if ((fp.force==FRF_RAT)||(fp.force==FRF_RAT_SDET))
         {
            l=fp.irat[0];
            rp=rat_parms(l);

            if (rp.degree==0)
               read_rat_parms(l);
         }
      }
   }
}


static void read_actions(void)
{
   int i,k,l,nact,*iact;
   hmc_parms_t hmc;
   action_parms_t ap;
   rat_parms_t rp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;

   for (i=0;i<nact;i++)
   {
      k=iact[i];
      ap=action_parms(k);

      if (ap.action==ACTIONS)
         read_action_parms(k);

      ap=action_parms(k);

      if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
      {
         l=ap.irat[0];
         rp=rat_parms(l);

         if (rp.degree==0)
            read_rat_parms(l);
      }
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
}


static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx,nsm;
   double kappa,mu,res,dtau;

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

   if (my_rank==0)
   {
      find_section("Deflation update scheme");
      read_line("dtau","%lf",&dtau);
      read_line("nsm","%d",&nsm);
   }

   MPI_Bcast(&dtau,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nsm,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_upd_parms(dtau,nsm);
}


static void read_solvers(void)
{
   int nact,*iact,nlv,nsp;
   int nfr,*ifr;
   int isap,idfl,i,j,k;
   hmc_parms_t hmc;
   mdint_parms_t mdp;
   action_parms_t ap;
   force_parms_t fp;
   solver_parms_t sp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   nlv=hmc.nlv;
   isap=0;
   idfl=0;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET)||
          (ap.action==ACF_TM2)||
          (ap.action==ACF_TM2_EO)||
          (ap.action==ACF_RAT)||
          (ap.action==ACF_RAT_SDET))
      {
         if ((ap.action==ACF_TM2)||(ap.action==ACF_TM2_EO)||
             (ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            nsp=2;
         else
            nsp=1;

         for (k=0;k<nsp;k++)
         {
            j=ap.isp[k];
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
   }

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);
      nfr=mdp.nfr;
      ifr=mdp.ifr;

      for (j=0;j<nfr;j++)
      {
         fp=force_parms(ifr[j]);

         if ((fp.force==FRF_TM1)||
             (fp.force==FRF_TM1_EO)||
             (fp.force==FRF_TM1_EO_SDET)||
             (fp.force==FRF_TM2)||
             (fp.force==FRF_TM2_EO)||
             (fp.force==FRF_RAT)||
             (fp.force==FRF_RAT_SDET))
         {
            k=fp.isp[0];
            sp=solver_parms(k);

            if (sp.solver==SOLVERS)
            {
               read_solver_parms(k);
               sp=solver_parms(k);

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
   }

   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();
}



static void check_files(void)
{
   int type,nion,nb,ib,n;
   int ns[4],bs[4];
   char *cnfg_dir;

   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;
   iodat.nb=0;
   iodat.ib=NPROC;

   if (type&0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,1,
            "check_files [check3.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);

      sprintf(line,"%s/%sn%d",cnfg_dir,nbase,first);
      lat_sizes(line,ns);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check3.c]","Lattice size mismatch");
   }
   else if (type&0x2)
   {
      error(name_size("%s/0/0/%sn%d_b0",cnfg_dir,nbase,first)>=NAME_SIZE,1,
            "check_files [check3.c]","cnfg_dir name is too long");
      sprintf(line,"%s/0/0",cnfg_dir);
      if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
         check_dir(line);

      sprintf(line,"%s/0/0/%sn%d_b0",cnfg_dir,nbase,first);
      blk_sizes(line,ns,bs);
      error_root((ns[0]!=N0)||(ns[1]!=N1)||(ns[2]!=N2)||(ns[3]!=N3),1,
                 "check_files [check3.c]","Lattice size mismatch");

      ib=blk_index(ns,bs,&nb);
      nion=iodat.nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [check3.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",cnfg_dir,nion-1,n-1,
                      nbase,last,nb-1)>=NAME_SIZE,1,
            "check_files [check3.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,ib/n,ib%n);
      strcpy(cnfg_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(cnfg_dir);

      iodat.nb=nb;
      iodat.ib=ib;
   }
   else if (type&0x4)
   {
      nion=iodat.nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [check3.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",cnfg_dir,nion-1,n-1,
                      nbase,last,NPROC-1)>=NAME_SIZE,1,
            "check_files [check3.c]","cnfg_dir name is too long");
      sprintf(line,"%s/%d/%d",cnfg_dir,my_rank/n,my_rank%n);
      strcpy(cnfg_dir,line);
      check_dir(cnfg_dir);
   }
}


static void set_fld(int icnfg)
{
   int type;
   double wt1,wt2;
   char *cnfg_dir;

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();
   type=iodat.type;
   cnfg_dir=iodat.cnfg_dir;

   if (type&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file,0x0);
   }
   else if (type&0x2)
   {
      set_nio_streams(iodat.nio_streams);
      sprintf(cnfg_file,"%s/%sn%d_b%d",cnfg_dir,nbase,icnfg,iodat.ib);
      blk_import_cnfg(cnfg_file,0x0);
   }
   else
   {
      set_nio_streams(iodat.nio_streams);
      sprintf(cnfg_file,"%s/%sn%d_%d",cnfg_dir,nbase,icnfg,my_rank);
      read_cnfg(cnfg_file);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Gauge field read from disk in %.2e sec\n\n",
             wt2-wt1);
      fflush(flog);
   }
}


static void set_nstep(int *nstep)
{
   int ilv,i;
   hmc_parms_t hmc;
   mdint_parms_t mdp;

   hmc=hmc_parms();
   ilv=hmc.nlv-1;
   mdp=mdint_parms(ilv);
   nstep[0]=mdp.nstep;

   for (i=1;i<4;i++)
   {
      if (mdp.integrator==OMF4)
         nstep[i]=2*nstep[i-1];
      else
      {
         if (i>=2)
            nstep[i]=2*nstep[i-2];
         else
            nstep[i]=(int)(sqrt(2.0)*(double)(nstep[i-1])+0.5);
      }
   }
}


static void reset_toplevel(int nstep)
{
   int ilv;
   hmc_parms_t hmc;
   mdint_parms_t mdp;

   hmc=hmc_parms();
   ilv=hmc.nlv-1;
   mdp=mdint_parms(ilv);
   set_mdint_parms(ilv,mdp.integrator,mdp.lambda,nstep,mdp.nfr,mdp.ifr);
}


static void chk_mode_regen(int isp,int *status)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if ((sp.solver==DFL_SAP_GCR)&&(status[2]>0))
      add2counter("modes",2,status+2);
}


static void start_hmc(qflt *act0,su3_dble *uold,su3_alg_dble *mold)
{
   int i,n,nact,*iact;
   int status[3];
   double *mu;
   su3_dble *udb;
   mdflds_t *mdfs;
   dfl_parms_t dfl;
   hmc_parms_t hmc;
   action_parms_t ap;

   clear_counters();

   udb=udfld();
   cm3x3_assign(4*VOLUME,udb,uold);
   set_ud_phase();
   random_mom();
   mdfs=mdflds();
   assign_alg2alg(4*VOLUME,(*mdfs).mom,mold);
   dfl=dfl_parms();

   if (dfl.Ns)
   {
      dfl_modes2(status);
      error_root((status[1]<0)||((status[1]==0)&&(status[0]<0)),1,
                 "start_hmc [check3.c]","Deflation subspace generation "
                 "failed (status = %d;%d)",status[0],status[1]);

      if (status[1]==0)
         add2counter("modes",0,status);
      else
         add2counter("modes",2,status+1);

      start_dfl_upd();
   }

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;
   n=2;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action==ACG)
         act0[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));

         if (ap.action==ACF_TM1)
            act0[n]=setpf1(mu[ap.imu[0]],ap.ipf,0);
         else if (ap.action==ACF_TM1_EO)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,0,0);
         else if (ap.action==ACF_TM1_EO_SDET)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,1,0);
         else if (ap.action==ACF_TM2)
         {
            status[2]=0;
            act0[n]=setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            chk_mode_regen(ap.isp[1],status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_TM2_EO)
         {
            status[2]=0;
            act0[n]=setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            chk_mode_regen(ap.isp[1],status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT)
         {
            status[2]=0;
            act0[n]=setpf3(ap.irat,ap.ipf,0,ap.isp[0],0,status);
            chk_mode_regen(ap.isp[0],status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT_SDET)
         {
            status[2]=0;
            act0[n]=setpf3(ap.irat,ap.ipf,1,ap.isp[0],0,status);
            chk_mode_regen(ap.isp[0],status);
            add2counter("field",ap.ipf,status);
         }
         else
            error_root(1,1,"start_hmc [check3.c]","Unknown action");

         n+=1;
      }
   }

   act0[0]=momentum_action(0);
}


static void end_hmc(qflt *act1)
{
   int i,n,ifr,nact,*iact;
   int status[3];
   double *mu;
   hmc_parms_t hmc;
   action_parms_t ap;
   force_parms_t fp;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;
   n=2;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);
      ifr=matching_force(iact[i]);
      fp=force_parms(ifr);

      if (ap.action==ACG)
         act1[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));
         status[2]=0;

         if (ap.action==ACF_TM1)
            act1[n]=action1(mu[ap.imu[0]],ap.ipf,ap.isp[0],fp.icr[0],
                            0,status);
         else if (ap.action==ACF_TM1_EO)
            act1[n]=action4(mu[ap.imu[0]],ap.ipf,0,ap.isp[0],fp.icr[0],
                            0,status);
         else if (ap.action==ACF_TM1_EO_SDET)
            act1[n]=action4(mu[ap.imu[0]],ap.ipf,1,ap.isp[0],fp.icr[0],
                            0,status);
         else if (ap.action==ACF_TM2)
            act1[n]=action2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                            fp.icr[0],0,status);
         else if (ap.action==ACF_TM2_EO)
            act1[n]=action5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[0],
                            fp.icr[0],0,status);
         else if (ap.action==ACF_RAT)
            act1[n]=action3(ap.irat,ap.ipf,0,ap.isp[0],0,status);
         else if (ap.action==ACF_RAT_SDET)
            act1[n]=action3(ap.irat,ap.ipf,1,ap.isp[0],0,status);

         chk_mode_regen(ap.isp[0],status);
         add2counter("action",iact[i],status);
         n+=1;
      }
   }

   act1[0]=momentum_action(0);
   unset_ud_phase();
}


static void restart_hmc(su3_dble *uold,su3_alg_dble *mold)
{
   int status[2];
   su3_dble *udb;
   mdflds_t *mdfs;
   dfl_parms_t dfl;

   clear_counters();

   udb=udfld();
   cm3x3_assign(4*VOLUME,uold,udb);
   set_ud_phase();
   set_flags(UPDATED_UD);

   mdfs=mdflds();
   assign_alg2alg(4*VOLUME,mold,(*mdfs).mom);

   dfl=dfl_parms();

   if (dfl.Ns)
   {
      dfl_modes2(status);
      error_root((status[1]<0)||((status[1]==0)&&(status[0]<0)),1,
                 "restart_hmc [check3.c]","Deflation subspace generation "
                 "failed (status = %d;%d)",status[0],status[1]);

      if (status[1]==0)
         add2counter("modes",0,status);
      else
         add2counter("modes",2,status+1);

      start_dfl_upd();
   }
}


static void sum_act(int nact,qflt *act0,qflt *act1,double *sm)
{
   int i;
   double *qsm[3];
   qflt act[3],da;

   for (i=0;i<3;i++)
   {
      act[i].q[0]=0.0;
      act[i].q[1]=0.0;
      qsm[i]=act[i].q;
   }

   for (i=0;i<=nact;i++)
   {
      add_qflt(act0[i].q,act[0].q,act[0].q);
      add_qflt(act1[i].q,act[1].q,act[1].q);
      da.q[0]=-act0[i].q[0];
      da.q[1]=-act0[i].q[1];
      add_qflt(act1[i].q,da.q,da.q);
      add_qflt(da.q,act[2].q,act[2].q);
   }

   global_qsum(3,qsm,qsm);

   for (i=0;i<3;i++)
      sm[i]=act[i].q[0];
}


int main(int argc,char *argv[])
{
   int icnfg,nact,i;
   int isap,idfl;
   int nwud,nws,nwv,nwvd;
   int nstep[4];
   double sm[3],dH[4];
   qflt *act0,*act1;
   su3_dble **usv;
   su3_alg_dble **fsv;
   hmc_parms_t hmc;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Conservation of the Hamilton function by the MD evolution\n");
      printf("---------------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   read_iodat();
   read_lat_parms();
   read_bc_parms();
   read_hmc_parms();
   read_actions();
   read_integrator();
   read_solvers();

   if (my_rank==0)
      fclose(fin);

   hmc_wsize(&nwud,&nws,&nwv,&nwvd);
   alloc_wud(nwud);
   alloc_ws(nws);
   wsd_uses_ws();
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   alloc_wfd(1);
   usv=reserve_wud(1);
   fsv=reserve_wfd(1);

   hmc=hmc_parms();
   nact=hmc.nact;
   act0=malloc(2*(nact+1)*sizeof(*act0));
   act1=act0+nact+1;
   error(act0==NULL,1,"main [check3.c]","Unable to allocate action arrays");
   set_nstep(nstep);

   check_machine();
   print_lat_parms();
   print_bc_parms(3);
   print_hmc_parms();
   print_action_parms();
   print_rat_parms();
   print_mdint_parms();
   print_force_parms2();
   print_solver_parms(&isap,&idfl);
   if (isap)
      print_sap_parms(0);
   if (idfl)
      print_dfl_parms(1);

   if (my_rank==0)
   {
      printf("Configuration storage type = ");

      if (iodat.type&0x1)
         printf("exported\n");
      else if (iodat.type&0x2)
         printf("block-exported\n");
      else
         printf("local\n");

      if (iodat.type&0x6)
         printf("Parallel configuration input: "
                "nio_nodes = %d, nio_streams = %d\n",
                iodat.nio_nodes,iodat.nio_streams);
      printf("\n");
      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   check_files();

   hmc_sanity_check();
   setup_counters();
   setup_chrono();
   print_msize(2);

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      if (my_rank==0)
      {
         printf("Configuration no %d\n\n",icnfg);
         fflush(flog);
      }

      set_fld(icnfg);

      for (i=0;i<4;i++)
      {
         if (i>0)
            reset_toplevel(nstep[i]);

         set_mdsteps();

         if (i==0)
            start_hmc(act0,usv[0],fsv[0]);
         else
            restart_hmc(usv[0],fsv[0]);

         run_mdint();
         end_hmc(act1);
         sum_act(nact,act0,act1,sm);
         dH[i]=fabs(sm[2]);

         if (my_rank==0)
         {
            if (i==0)
            {
               printf("start_hmc:\n");
               printf("H = %.6e\n",sm[0]);
               fflush(flog);
            }

            printf("run_md:\n");
            printf("nstep = %d, tau/nstep = %.2e\n",
                   nstep[i],hmc.tau/(double)(nstep[i]));
            printf("H = %.6e, |dH| = %.2e\n",sm[1],dH[i]);
            fflush(flog);
         }

         print_all_avgstat();
      }

      if (my_rank==0)
      {
         printf("\n");
         printf("nstep[0] = %d,            |dH| = %.2e\n",nstep[0],dH[0]);

         for (i=1;i<4;i++)
         {
            printf("nstep[i-1]/nstep[i] = %.2e, ",
                   (double)(nstep[i-1])/(double)(nstep[i]));
            printf("|dH| = %.2e, |dH[i]|/|dH[i-1]| = %.2e\n",
                   dH[i],dH[i]/dH[i-1]);
         }

         printf("\n");
         fflush(flog);
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
