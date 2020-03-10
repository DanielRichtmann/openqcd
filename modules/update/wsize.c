
/*******************************************************************************
*
* File wsize.c
*
* Copyright (C) 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Workspaces required for the simulations.
*
*   void hmc_wsize(int *nwud,int *nws,int *nwv,int *nwvd)
*     Determines the minimal sizes of the workspaces required for the
*     HMC algorithm.
*
*   void smd_wsize(int *nwud,int *nwfd,int *nws,int *nwv,int *nwvd)
*     Determines the minimal sizes of the workspaces required for the
*     SMD algorithm.
*
* These programs consult the parameter data base and determine how many
* fields of the various types must be allocated, using the appropriate
* workspace programs, to be able to run a simulation. The numbers of
* fields determined are:
*
*  nwud     Double-precision gauge fields.
*
*  nwfd     Double-precision force fields.
*
*  nws      Single-precision spinor fields.
*
*  nwv      Single-precision complex vector fields.
*
*  nwvd     Double-precision complex vector fields.
*
* These names match the alloc_wud(),..,alloc_wvd() functions in the module
* utils/wspace.c. The number nws of single-precision spinor fields assumes
* that the workspace for spinor fields is allocated by calling alloc_ws(nws)
* and wsd_uses_ws() so that single- and double-precision fields share the
* same workspace.
*
* The programs in this module do not perform any global communications and
* return the same values on all MPI processes.
*
*******************************************************************************/

#define WSIZE_C

#include <stdlib.h>
#include <stdio.h>
#include "flags.h"
#include "utils.h"
#include "update.h"


static void maxn(int *n,int m)
{
   if ((*n)<m)
      (*n)=m;
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();

   maxn(nws,dp.Ns+2);
   maxn(nwv,2*dpp.nkv+2);
   maxn(nwvd,4);
}


static void solver_wsize(int isp,int nsds,int np,int *nws,int *nwv,int *nwvd)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if (sp.solver==CGNE)
      maxn(nws,nsds+11);
   else if (sp.solver==MSCG)
   {
      if (np>1)
         maxn(nws,nsds+2*np+6);
      else
         maxn(nws,nsds+10);
   }
   else if (sp.solver==SAP_GCR)
      maxn(nws,nsds+2*sp.nkv+5);
   else if (sp.solver==DFL_SAP_GCR)
   {
      maxn(nws,nsds+2*sp.nkv+6);
      dfl_wsize(nws,nwv,nwvd);
   }
}


static void action_wsize(int nact,int *iact,int *nws,int *nwv,int *nwvd)
{
   int nsds,np,i;
   action_parms_t ap;
   solver_parms_t sp;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if ((ap.action==ACF_TM1)||
          (ap.action==ACF_TM1_EO)||
          (ap.action==ACF_TM1_EO_SDET))
      {
         nsds=2;
         solver_wsize(ap.isp[0],nsds,0,nws,nwv,nwvd);
         maxn(nws,4);
      }
      else if ((ap.action==ACF_TM2)||
               (ap.action==ACF_TM2_EO))
      {
         nsds=2;
         solver_wsize(ap.isp[0],nsds,0,nws,nwv,nwvd);
         solver_wsize(ap.isp[1],nsds,0,nws,nwv,nwvd);
      }
      else if ((ap.action==ACF_RAT)||
               (ap.action==ACF_RAT_SDET))
      {
         np=ap.irat[2]-ap.irat[1]+1;
         sp=solver_parms(ap.isp[0]);

         if (sp.solver==MSCG)
            nsds=2*np+2;
         else
            nsds=4;

         solver_wsize(ap.isp[0],nsds,np,nws,nwv,nwvd);
      }
   }
}


static void force_wsize(int nlv,int *nws,int *nwv,int *nwvd)
{
   int nfr,*ifr;
   int nsds,np,i,j;
   mdint_parms_t mdp;
   force_parms_t fp;
   solver_parms_t sp;

   for (i=0;i<nlv;i++)
   {
      mdp=mdint_parms(i);
      nfr=mdp.nfr;
      ifr=mdp.ifr;

      for (j=0;j<nfr;j++)
      {
         fp=force_parms(ifr[j]);

         if ((fp.force==FRF_TM1)||
             (fp.force==FRF_TM2))
         {
            sp=solver_parms(fp.isp[0]);
            nsds=4;

            if (fp.icr[0])
            {
               if (sp.solver==CGNE)
                  nsds+=2;
               else
                  nsds+=4;
            }

            solver_wsize(fp.isp[0],nsds,0,nws,nwv,nwvd);
         }
         else if ((fp.force==FRF_TM1_EO)||
                  (fp.force==FRF_TM1_EO_SDET)||
                  (fp.force==FRF_TM2_EO))
         {
            sp=solver_parms(fp.isp[0]);

            if (sp.solver==CGNE)
               nsds=4;
            else
               nsds=6;

            if (fp.icr[0])
               nsds+=2;

            solver_wsize(fp.isp[0],nsds,0,nws,nwv,nwvd);
         }
         else if ((fp.force==FRF_RAT)||
                  (fp.force==FRF_RAT_SDET))
         {
            np=fp.irat[2]-fp.irat[1]+1;
            sp=solver_parms(fp.isp[0]);

            if (sp.solver==MSCG)
               nsds=2*np+2;
            else
               nsds=6;

            solver_wsize(fp.isp[0],nsds,np,nws,nwv,nwvd);
         }
      }
   }
}


void hmc_wsize(int *nwud,int *nws,int *nwv,int *nwvd)
{
   hmc_parms_t hmc;

   hmc=hmc_parms();

   (*nwud)=1;
   (*nws)=0;
   (*nwv)=0;
   (*nwvd)=0;

   action_wsize(hmc.nact,hmc.iact,nws,nwv,nwvd);
   force_wsize(hmc.nlv,nws,nwv,nwvd);
}


void smd_wsize(int *nwud,int *nwfd,int *nws,int *nwv,int *nwvd)
{
   smd_parms_t smd;

   smd=smd_parms();

   if (smd.iacc)
   {
      (*nwud)=1;
      (*nwfd)=1;
   }
   else
   {
      (*nwud)=0;
      (*nwfd)=0;
   }

   (*nws)=0;
   (*nwv)=0;
   (*nwvd)=0;

   action_wsize(smd.nact,smd.iact,nws,nwv,nwvd);
   force_wsize(smd.nlv,nws,nwv,nwvd);
}
