
/*******************************************************************************
*
* File hmc.c
*
* Copyright (C) 2005, 2007, 2009-2013,    Martin Luescher, Filippo Palombi,
*               2016-2018                 Stefan Schaefer, Isabel Campos
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* HMC simulation algorithm.
*
*   int run_hmc(qflt *act0,qflt *act1)
*     Applies a complete HMC update cycle to the current gauge field. On
*     exit the arrays act0 and act1 contain the local parts of the momentum,
*     gauge and pseudo-fermion actions at the beginning and the end of the
*     molecular-dynamics evolution. The program returns 1 or 0 depending
*     on whether the configuration was accepted or not.
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The elements of the action arrays act0 and act1 are
*
*  act[0]        action of the momentum field,
*  act[1]        gauge field action,
*  act[2+n]      pseudo-fermion action number n,
*
* where the pseudo-fermion actions are counted from n=0 in steps of 1 as
* they appear in the action array hmc.iact in the structure hmc returned
* by hmc_parms(). The arrays must thus have at least hmc.nact+1 elements.
*
* The HMC algorithm is implemented as specified by the parameter data base.
* Accepted new gauge field configurations are renormalized to SU(3) on all
* active links. If a configuration is rejected, the field is reset to the
* value it had at the beginning of the molecular-dynamics evolution.
*
* The program run_hmc() takes care of the counters, the chronological solver,
* the deflation subspace and the phase-setting [uflds/uflds.c]. Along the
* molecular-dynamics trajectory the deflation subspace is updated according
* to the parameter data base [flags/dfl_parms.c]. On exit the gauge field is
* phase-unset.
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define HMC_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "lattice.h"
#include "utils.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "dfl.h"
#include "forces.h"
#include "update.h"
#include "global.h"

static su3_dble ubnd[3] ALIGNED16;


static void chk_mode_regen(int isp,int *status)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if ((sp.solver==DFL_SAP_GCR)&&(status[2]>0))
      add2counter("modes",2,status+2);
}


static void start_hmc(qflt *act0,su3_dble *uold)
{
   int i,n,nact,*iact;
   int status[3];
   double *mu;
   su3_dble *udb;
   dfl_parms_t dfl;
   hmc_parms_t hmc;
   action_parms_t ap;

   hmc=hmc_parms();
   nact=hmc.nact;
   iact=hmc.iact;
   mu=hmc.mu;

   random_mom();
   unset_ud_phase();
   udb=udfld();
   cm3x3_assign(4*VOLUME,udb,uold);
   if ((cpr[0]==(NPROC0-1))&&((bc_type()==1)||(bc_type()==2)))
      cm3x3_assign(3,udb+4*VOLUME+7*(BNDRY/4),ubnd);

   if (hmc.npf)
   {
      clear_counters();
      reset_chrono(0);
      set_ud_phase();
      dfl=dfl_parms();

      if (dfl.Ns)
      {
         dfl_modes2(status);
         error_root((status[1]<0)||((status[1]==0)&&(status[0]<0)),1,
                    "start_hmc [hmc.c]","Deflation subspace generation "
                    "failed (status = %d;%d)",status[0],status[1]);

         if (status[1]==0)
            add2counter("modes",0,status);
         else
            add2counter("modes",2,status+1);

         start_dfl_upd();
      }
   }

   act0[0]=momentum_action(0);
   n=2;

   for (i=0;i<nact;i++)
   {
      ap=action_parms(iact[i]);

      if (ap.action==ACG)
         act0[1]=action0(0);
      else
      {
         set_sw_parms(sea_quark_mass(ap.im0));
         status[2]=0;

         if (ap.action==ACF_TM1)
            act0[n]=setpf1(mu[ap.imu[0]],ap.ipf,0);
         else if (ap.action==ACF_TM1_EO)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,0,0);
         else if (ap.action==ACF_TM1_EO_SDET)
            act0[n]=setpf4(mu[ap.imu[0]],ap.ipf,1,0);
         else if (ap.action==ACF_TM2)
         {
            act0[n]=setpf2(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            chk_mode_regen(ap.isp[1],status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_TM2_EO)
         {
            act0[n]=setpf5(mu[ap.imu[0]],mu[ap.imu[1]],ap.ipf,ap.isp[1],
                           0,status);
            chk_mode_regen(ap.isp[1],status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT)
         {
            act0[n]=setpf3(ap.irat,ap.ipf,0,ap.isp[1],0,status);
            chk_mode_regen(ap.isp[1],status);
            add2counter("field",ap.ipf,status);
         }
         else if (ap.action==ACF_RAT_SDET)
         {
            act0[n]=setpf3(ap.irat,ap.ipf,1,ap.isp[1],0,status);
            chk_mode_regen(ap.isp[1],status);
            add2counter("field",ap.ipf,status);
         }
         else
            error(1,1,"start_hmc [hmc.c]","Unknown action");

         n+=1;
      }
   }
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

   act1[0]=momentum_action(0);
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
         else
            error(1,1,"end_hmc [hmc.c]","Unknown action");

         chk_mode_regen(ap.isp[0],status);
         add2counter("action",iact[i],status);
         n+=1;
      }
   }
}


static int accept_hmc(qflt *act0,qflt *act1,su3_dble *uold)
{
   int my_rank,iac,i;
   double r,da,*qsm[1];
   qflt act,dact;
   su3_dble *udb;
   hmc_parms_t hmc;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   hmc=hmc_parms();
   iac=0;
   dact.q[0]=0.0;
   dact.q[1]=0.0;

   for (i=0;i<=hmc.nact;i++)
   {
      act.q[0]=-act0[i].q[0];
      act.q[1]=-act0[i].q[1];
      add_qflt(act1[i].q,act.q,act.q);
      add_qflt(act.q,dact.q,dact.q);
   }

   if (NPROC>1)
   {
      qsm[0]=dact.q;
      global_qsum(1,qsm,qsm);
   }

   if (my_rank==0)
   {
      ranlxd(&r,1);
      da=dact.q[0];

      if (da<=0.0)
         iac=1;
      else if (r<=exp(-da))
         iac=1;
   }

   if (NPROC>1)
      MPI_Bcast(&iac,1,MPI_INT,0,MPI_COMM_WORLD);

   if (iac==0)
   {
      udb=udfld();
      cm3x3_assign(4*VOLUME,uold,udb);
      if ((cpr[0]==(NPROC0-1))&&((bc_type()==1)||(bc_type()==2)))
         cm3x3_assign(3,ubnd,udb+4*VOLUME+7*(BNDRY/4));
      set_flags(UPDATED_UD);
      set_flags(UNSET_UD_PHASE);
   }
   else
   {
      unset_ud_phase();
      renormalize_ud();
   }

   return iac;
}


int run_hmc(qflt *act0,qflt *act1)
{
   int iac;
   su3_dble **uold;

   uold=reserve_wud(1);

   start_hmc(act0,uold[0]);
   run_mdint();
   end_hmc(act1);
   iac=accept_hmc(act0,act1,uold[0]);

   release_wud();

   return iac;
}
