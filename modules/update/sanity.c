
/*******************************************************************************
*
* File sanity.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Simulation parameter sanity checks.
*
*   void hmc_sanity_check(void)
*     Checks the chosen parameters for the HMC algorithm and terminates
*     with an error message if an inconsistency is discovered.
*
*   void smd_sanity_check(void)
*     Checks the chosen parameters for the SMD algorithm and terminates
*     with an error message if an inconsistency is discovered.
*
*   int matching_force(int iact)
*     Returns the index in the parameter data base of the force that
*     corresponds to the action with index iact. An error occurs if
*     no matching force is found or if iact is out of range.
*
* The programs in this module do not perform any global communications and
* can be locally called, but the sanity checks are only performed on the MPI
* process with rank 0.
*
* It is assumed that these programs are called only after all action, force
* and simulation parameters are set.
*
*******************************************************************************/

#define SANITY_C

#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "update.h"
#include "global.h"

static int nrs=0,*rs;
static int nacm=0,*ifrc;


static void init_rs(int nr)
{
   int k;

   if (nr>nrs)
   {
      if (nrs>0)
         free(rs);

      rs=malloc(nr*sizeof(*rs));
      error_root(rs==NULL,1,"init_rs [sanity.c]",
                 "Unable to allocate auxiliary array");
      nrs=nr;
   }

   for (k=0;k<nr;k++)
      rs[k]=0;
}


static int check_rat_actions(int nact,int *iact)
{
   int k,l,j,ie;
   int ir,nr,im0,isw;
   action_parms_t ap;
   rat_parms_t rp;
   sw_parms_t swp;

   ie=0;

   for (k=0;k<nact;k++)
   {
      ap=action_parms(iact[k]);

      if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
      {
         ir=ap.irat[0];
         im0=ap.im0;
         rp=rat_parms(ir);
         nr=rp.degree;
         init_rs(nr);
         swp=sw_parms();
         isw=0;

         for (l=0;l<nact;l++)
         {
            ap=action_parms(iact[l]);

            if ((ap.action==ACF_RAT)||(ap.action==ACF_RAT_SDET))
            {
               if ((ap.irat[0]==ir)&&(ap.im0==im0))
               {
                  if (ap.action==ACF_RAT_SDET)
                     isw+=1;

                  for (j=ap.irat[1];j<=ap.irat[2];j++)
                     rs[j]+=1;
               }
            }
         }

         if (swp.isw)
         {
            for (l=0;l<nr;l++)
               ie|=(rs[l]!=rs[0]);
         }
         else
         {
            for (l=0;l<nr;l++)
               ie|=(rs[l]!=isw);
         }
      }
   }

   return ie;
}


static int match_force(action_parms_t ap,force_parms_t fp)
{
   int ie;

   ie=1;

   if (ap.action==ACG)
      ie&=(fp.force==FRG);
   else if (ap.action==ACF_TM1)
   {
      ie&=(fp.force==FRF_TM1);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);
   }
   else if (ap.action==ACF_TM1_EO)
   {
      ie&=(fp.force==FRF_TM1_EO);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);
   }
   else if (ap.action==ACF_TM1_EO_SDET)
   {
      ie&=(fp.force==FRF_TM1_EO_SDET);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);
   }
   else if (ap.action==ACF_TM2)
   {
      ie&=(fp.force==FRF_TM2);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);
      ie&=(ap.imu[1]==fp.imu[1]);
   }
   else if (ap.action==ACF_TM2_EO)
   {
      ie&=(fp.force==FRF_TM2_EO);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.imu[0]==fp.imu[0]);
      ie&=(ap.imu[1]==fp.imu[1]);
   }
   else if (ap.action==ACF_RAT)
   {
      ie&=(fp.force==FRF_RAT);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.irat[0]==fp.irat[0]);
      ie&=(ap.irat[1]==fp.irat[1]);
      ie&=(ap.irat[2]==fp.irat[2]);
   }
   else if (ap.action==ACF_RAT_SDET)
   {
      ie&=(fp.force==FRF_RAT_SDET);
      ie&=(ap.ipf==fp.ipf);
      ie&=(ap.im0==fp.im0);
      ie&=(ap.irat[0]==fp.irat[0]);
      ie&=(ap.irat[1]==fp.irat[1]);
      ie&=(ap.irat[2]==fp.irat[2]);
   }
   else
      ie=0;

   return ie;
}


static void check_all_actions(int nlv,int nact,int *iact,int npf,int nmu)
{
   int iacg,ierat,iepf,iemu,iem0;
   int i,j,nfr,*ifr;
   action_parms_t ap;
   rat_parms_t rp;
   mdint_parms_t mdp;
   force_parms_t fp;

   iacg=0;
   ierat=0;
   iepf=0;
   iemu=0;
   iem0=0;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action==ACG)
         iacg+=1;
      else if ((ap.action==ACF_TM1)||
               (ap.action==ACF_TM1_EO)||
               (ap.action==ACF_TM1_EO_SDET)||
               (ap.action==ACF_TM2)||
               (ap.action==ACF_TM2_EO))
      {
         iepf|=(ap.ipf<0);
         iepf|=(ap.ipf>=npf);
         iemu|=(ap.imu[0]<0);
         iemu|=(ap.imu[0]>=nmu);
         iem0|=(sea_quark_mass(ap.im0)==DBL_MAX);

         if ((ap.action==ACF_TM2)||
             (ap.action==ACF_TM2_EO))
         {
            iemu|=(ap.imu[1]<0);
            iemu|=(ap.imu[1]>=nmu);
         }
      }
      else if ((ap.action==ACF_RAT)||
               (ap.action==ACF_RAT_SDET))
      {
         iepf|=(ap.ipf<0);
         iepf|=(ap.ipf>=npf);
         iem0|=(sea_quark_mass(ap.im0)==DBL_MAX);

         rp=rat_parms(ap.irat[0]);
         ierat|=(rp.degree==0);
         ierat|=(ap.irat[1]<0);
         ierat|=(ap.irat[2]<ap.irat[1]);
         ierat|=(ap.irat[2]>=rp.degree);
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
             (fp.force==FRF_TM2_EO))
         {
            iepf|=(fp.ipf<0);
            iepf|=(fp.ipf>=npf);
            iemu|=(fp.imu[0]<0);
            iemu|=(fp.imu[0]>=nmu);
            iem0|=(sea_quark_mass(fp.im0)==DBL_MAX);

            if ((fp.force==FRF_TM2)||
                (fp.force==FRF_TM2_EO))
            {
               iemu|=(fp.imu[1]<0);
               iemu|=(fp.imu[1]>=nmu);
            }
         }
         else if ((fp.force==FRF_RAT)||
                  (fp.force==FRF_RAT_SDET))
         {
            iepf|=(fp.ipf<0);
            iepf|=(fp.ipf>=npf);
            iem0|=(sea_quark_mass(fp.im0)==DBL_MAX);

            rp=rat_parms(fp.irat[0]);
            ierat|=(rp.degree==0);
            ierat|=(fp.irat[1]<0);
            ierat|=(fp.irat[2]<fp.irat[1]);
            ierat|=(fp.irat[2]>=rp.degree);
         }
      }
   }

   if (ierat==0)
      ierat|=check_rat_actions(nact,iact);

   error_root(iacg!=1,1,"check_all_actions [sanity.c]",
              "Gauge action is missing or occurs several times");
   error_root(ierat!=0,1,"check_all_actions [sanity.c]",
              "Some rational functions are not or not correctly specified");
   error_root(iepf!=0,1,"check_all_actions [sanity.c]",
              "Some pseudo-fermion indices are out of range");
   error_root(iemu!=0,1,"check_all_actions [sanity.c]",
              "Some twisted-mass indices are out of range");
   error_root(iem0!=0,1,"check_all_actions [sanity.c]",
              "Some sea-quark mass indices are out of range");
}


static void check_all_forces(int nlv,int nact,int *iact)
{
   int ie,ic;
   int i,j,k,nfr,*ifr;
   action_parms_t ap;
   mdint_parms_t mdp;
   force_parms_t fp;

   ie=0;

   for (k=0;k<nact;k++)
   {
      ap=action_parms(iact[k]);
      ic=0;

      for (i=0;i<nlv;i++)
      {
         mdp=mdint_parms(i);
         nfr=mdp.nfr;
         ifr=mdp.ifr;

         for (j=0;j<nfr;j++)
         {
            fp=force_parms(ifr[j]);
            ic+=match_force(ap,fp);
         }
      }

      ie|=(ic!=1);
   }

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);
      nfr=mdp.nfr;
      ifr=mdp.ifr;

      for (j=0;j<nfr;j++)
      {
         fp=force_parms(ifr[j]);
         ic=0;

         for (k=0;k<nact;k++)
         {
            ap=action_parms(iact[k]);
            ic+=match_force(ap,fp);
         }

         ie|=(ic!=1);
      }
   }

   error_root(ie!=0,1,"check_all_forces [sanity.c]",
              "Specified actions and forces do not match");
}


static void set_ifrc(int nlv,int nact,int *iact)
{
   int i,j,k,iak;
   int nfr,*ifr;
   action_parms_t ap;
   force_parms_t fp;
   mdint_parms_t mdp;

   nacm=0;

   for (k=0;k<nact;k++)
   {
      if (nacm<iact[k])
         nacm=iact[k];
   }

   nacm+=1;
   ifrc=malloc(nacm*sizeof(*ifrc));
   error_loc(ifrc==NULL,1,"set_ifrc [sanity.c]",
             "Unable to allocate index array");

   for (k=0;k<nacm;k++)
      ifrc[k]=-1;

   for (k=0;k<nact;k++)
   {
      iak=iact[k];
      ap=action_parms(iak);

      for (i=0;(i<nlv)&&(ifrc[iak]==-1);i++)
      {
         mdp=mdint_parms(i);
         nfr=mdp.nfr;
         ifr=mdp.ifr;

         for (j=0;(j<nfr)&&(ifrc[iak]==-1);j++)
         {
            fp=force_parms(ifr[j]);

            if (match_force(ap,fp))
               ifrc[iak]=ifr[j];
         }
      }
   }
}


void hmc_sanity_check(void)
{
   int my_rank;
   int nlv,nact,*iact;
   int npf,nmu;
   hmc_parms_t hmc;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      hmc=hmc_parms();
      nlv=hmc.nlv;
      nact=hmc.nact;
      iact=hmc.iact;
      npf=hmc.npf;
      nmu=hmc.nmu;

      error_root((nlv<1)||(nact<1),1,"hmc_sanity_check [sanity.c]",
                 "No of actions and integration levels must be at least 1");

      check_all_actions(nlv,nact,iact,npf,nmu);
      check_all_forces(nlv,nact,iact);
   }
}


void smd_sanity_check(void)
{
   int my_rank;
   int nlv,nact,*iact;
   int npf,nmu;
   smd_parms_t smd;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      smd=smd_parms();
      nlv=smd.nlv;
      nact=smd.nact;
      iact=smd.iact;
      npf=smd.npf;
      nmu=smd.nmu;

      error_root((nlv<1)||(nact<1),1,"smd_sanity_check [sanity.c]",
                 "No of actions and integration levels must be at least 1");

      check_all_actions(nlv,nact,iact,npf,nmu);
      check_all_forces(nlv,nact,iact);
   }
}


int matching_force(int iact)
{
   int ifr;
   hmc_parms_t hmc;
   smd_parms_t smd;

   if (nacm==0)
   {
      hmc=hmc_parms();
      smd=smd_parms();

      if (hmc.nlv>0)
         set_ifrc(hmc.nlv,hmc.nact,hmc.iact);
      else if (smd.nlv>0)
         set_ifrc(smd.nlv,smd.nact,smd.iact);
      else
         error_loc(1,1,"matching_force [sanity.c]",
                   "Simulation parameters are not set");
   }

   if ((iact>=0)&&(iact<nacm))
      ifr=ifrc[iact];
   else
      ifr=-1;

   error_loc(ifr<0,1,"matching_force [sanity.c]","Matching force not found");

   return ifr;
}
