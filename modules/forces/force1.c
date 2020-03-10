
/*******************************************************************************
*
* File force1.c
*
* Copyright (C) 2011-2013, 2017, 2018  Stefan Schaefer, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted mass pseudo-fermion action and force.
*
*   qflt setpf1(double mu,int ipf,int icom)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf (see the notes).
*
*   void rotpf1(double mu,int ipf,int isp,int icr,double c1,double c2,
*               int *status)
*     Generates a pseudo-fermion field eta with probability proportional
*     to exp(-Spf) and and replaces phi by c1*phi+c2*eta (see the notes).
*     The chronological solver stack with index icr is updated as well.
*
*   void force1(double mu,int ipf,int isp,int icr,double c,int *status)
*     Computes the force deriving from the action Spf (see the notes).
*     The calculated force is multiplied by c and added to the molecular-
*     dynamics force field.
*
*   qflt action1(double mu,int ipf,int isp,int icr,int icom,int *status)
*     Returns the action Spf (see the notes).
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,(Dw^dag*Dw+mu^2)^(-1)*phi),
*
* where Dw denotes the (improved) Wilson-Dirac operator and phi the pseudo-
* fermion field.
*
* The common parameters of the programs in this module are:
*
*   mu            Twisted mass parameter in Spf.
*
*   ipf           Index of the pseudo-fermion field phi in the
*                 structure returned by mdflds() [mdflds.c].
*
*   isp           Index of the solver parameter set that describes
*                 the solver to be used for the solution of the
*                 Dirac equation.
*
*   icr           Index of the chronological solver stack to use.
*                 Setting icr=0 disables this feature.
*
*   icom          The action returned by the programs setpf1() and
*                 action1() is summed over all MPI processes if icom=1.
*                 Otherwise the local part of the action is returned.
*
*   status        Status values returned by the solver used for the
*                 solution of the Dirac equation.
*
* The supported solvers are CGNE, SAP_GCR and DFL_SAP_GCR. Depending
* on the program and the solver, the number of status variables is:
*
*                  CGNE         SAP_GCR       DFL_SAP_GCR
*   rotpf1()      (icr>0)       (icr>0)        3*(icr>0)
*   force1()         1             2               6
*   action1()        1             1               3
*
* Note that, in force1(), the GCR solvers solve the Dirac equations twice.
* In these cases, the program writes the status values one after the other
* to the array. The bare quark mass m0 is the one last set by sw_parms()
* [flags/lat_parms.c] and it is taken for granted that the parameters of
* the solver have been set by set_solver_parms() [flags/solver_parms.c].
*
* The programs rotpf1() and force1() attempt to propagate the solutions of
* the Dirac equation along the molecular-dynamics trajectories, using the
* field stack number icr (no fields are propagated if icr=0). This feature
* assumes the program setup_chrono() [update/chrono.c] is called before
* rotpf1() or force1() is called for the first time.
*
* The required workspaces of double-precision spinor fields are
*
*   setpf1()         2
*
* and
*
*                  CGNE         SAP_GCR       DFL_SAP_GCR
*   rotpf1()         2             2               2
*   force1()     2+(icr>0)     2+2*(icr>0)     2+2*(icr>0)
*   action1()        2             2               2
*
* (these figures do not include the workspace required by the solvers).
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define FORCE1_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "sw_term.h"
#include "sflds.h"
#include "dirac.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "update.h"
#include "forces.h"
#include "global.h"

#if (defined FORCE_DBG)

static char *program[3]={"action1","force1","rotpf1"};


static void solver_info(int ipgm,int icr,double mu,solver_parms_t sp)
{
   if (sp.solver==CGNE)
      message("[%s]: CGNE solver, icr = %d, istop = %d, mu = %.2e\n",
              program[ipgm],icr,sp.istop,mu);
   else if (sp.solver==SAP_GCR)
      message("[%s]: SAP_GCR solver, icr = %d, istop = %d, mu = %.2e\n",
              program[ipgm],icr,sp.istop,mu);
   else if (sp.solver==DFL_SAP_GCR)
      message("[%s]: DFL_SAP_GCR solver, icr = %d, istop = %d, mu = %.2e\n",
              program[ipgm],icr,sp.istop,mu);
}


static void check_flds0(double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   Dw_dble(mu,psi,rho);
   mulg5_dble(VOLUME,rho);
   mulr_spinor_add_dble(VOLUME,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME,1,rho);
      rpsi/=unorm_dble(VOLUME,1,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME,1,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME,1,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[action1]: Residue of psi = %.1e (should be <= %.1e)\n",
           rpsi,sp.res);

   release_wsd();
}


static void check_flds1(double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi,spinor_dble *chi)
{
   double rchi,rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   Dw_dble(-mu,chi,rho);
   mulg5_dble(VOLUME,rho);
   mulr_spinor_add_dble(VOLUME,rho,psi,-1.0);

   if (sp.istop)
   {
      rchi=unorm_dble(VOLUME,1,rho);
      rchi/=unorm_dble(VOLUME,1,psi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME,1,rho);
      rchi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME,1,psi);
      rchi/=qnrm.q[0];
      rchi=sqrt(rchi);
   }

   Dw_dble(mu,psi,rho);
   mulg5_dble(VOLUME,rho);
   mulr_spinor_add_dble(VOLUME,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME,1,rho);
      rpsi/=unorm_dble(VOLUME,1,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME,1,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME,1,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[force1]: Residue of psi = %.1e (should be <= %.1e)\n",
           rpsi,sp.res);
   message("[force1]: Residue of chi = %.1e (should be <= %.1e)\n",
           rchi,sp.res);

   release_wsd();
}


static void check_flds2(double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   Dw_dble(-mu,psi,rho);
   mulg5_dble(VOLUME,rho);
   Dw_dble(mu,rho,psi);
   mulg5_dble(VOLUME,psi);
   mulr_spinor_add_dble(VOLUME,psi,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME,1,psi);
      rpsi/=unorm_dble(VOLUME,1,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME,1,psi);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME,1,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[rotpf1]: Residue of psi = %.1e\n",rpsi);

   release_wsd();
}

#endif

qflt setpf1(double mu,int ipf,int icom)
{
   qflt act;
   spinor_dble *phi,*eta,*psi,**wsd;
   mdflds_t *mdfs;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   eta=wsd[0];
   psi=wsd[1];

   random_sd(VOLUME,eta,1.0);
   bnd_sd2zero(ALL_PTS,eta);
   act=norm_square_dble(VOLUME,icom,eta);
   sw_term(NO_PTS);
   Dw_dble(mu,eta,psi);
   mulg5_dble(VOLUME,psi);
   assign_sd2sd(VOLUME,psi,phi);

   release_wsd();

   return act;
}


void rotpf1(double mu,int ipf,int isp,int icr,double c1,double c2,
            int *status)
{
   spinor_dble *phi,*eta,*psi,**wsd;
   spinor_dble *rho,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   eta=wsd[0];
   psi=wsd[1];

   random_sd(VOLUME,eta,1.0);
   bnd_sd2zero(ALL_PTS,eta);
   sw_term(NO_PTS);
   Dw_dble(mu,eta,psi);
   mulg5_dble(VOLUME,psi);
   combine_spinor_dble(VOLUME,phi,psi,c1,c2);

   if (get_chrono(icr,psi))
   {
      sp=solver_parms(isp);

#if (defined FORCE_DBG)
      solver_info(2,icr,mu,sp);
#endif

      if (sp.solver==CGNE)
      {
         tmcg(sp.nmx,sp.istop,sp.res,mu,eta,eta,status);

         error_root(status[0]<0,1,"rotpf1 [force1.c]",
                    "CGNE solver failed (mu = %.4e, parameter set no %d, "
                    "status = %d)",mu,isp,status[0]);

         rsd=reserve_wsd(1);
         rho=rsd[0];
         Dw_dble(mu,eta,rho);
         mulg5_dble(VOLUME,rho);
         combine_spinor_dble(VOLUME,psi,rho,c1,c2);
         release_wsd();
      }
      else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      {
         sap=sap_parms();
         set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
         mulg5_dble(VOLUME,eta);

         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,eta,
                    eta,status);

            error_root(status[0]<0,1,"rotpf1 [force1.c]",
                       "SAP_GCR solver failed (mu = %.4e, parameter set no %d, "
                       "status = %d)",mu,isp,status[0]);
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,eta,
                         eta,status);

            error_root((status[0]<0)||(status[1]<0),1,"rotpf1 [force1.c]",
                       "DFL_SAP_GCR solver failed (mu = %.4e, "
                       "parameter set no %d, status = %d,%d,%d",
                       mu,isp,status[0],status[1],status[2]);
         }

         combine_spinor_dble(VOLUME,psi,eta,c1,c2);
      }
      else
         error(1,1,"rotpf1 [force1.c]","Unknown solver");

      add_chrono(icr,psi);

#if (defined FORCE_DBG)
      check_flds2(mu,sp,phi,psi);
#endif
   }
   else if (icr>0)
   {
      sp=solver_parms(isp);

      if (sp.solver==DFL_SAP_GCR)
      {
         status[0]=0;
         status[1]=0;
         status[2]=0;
      }
      else
         status[0]=0;
   }

   release_wsd();
}


void force1(double mu,int ipf,int isp,int icr,double c,int *status)
{
   int l;
   double res0,res1;
   qflt rqsm;
   spinor_dble *phi,*chi,*psi,**wsd;
   spinor_dble *rho,*eta,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];
   chi=wsd[1];

   sw_term(NO_PTS);
   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(1,icr,mu,sp);
#endif

   if (sp.solver==CGNE)
   {
      status[0]=0;

      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(1);
         rho=rsd[0];

         Dw_dble(-mu,chi,psi);
         mulg5_dble(VOLUME,psi);
         Dw_dble(mu,psi,rho);
         mulg5_dble(VOLUME,rho);
         mulr_spinor_add_dble(VOLUME,rho,phi,-1.0);

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME,1,phi);
            res1=unorm_dble(VOLUME,1,rho)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME,1,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME,1,rho);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcg(sp.nmx,sp.istop,sp.res/res1,mu,rho,psi,status);
               mulr_spinor_add_dble(VOLUME,chi,psi,-1.0);
            }
         }
         else
         {
            assign_sd2sd(VOLUME,phi,psi);
            tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,status);
         }

         release_wsd();
      }
      else
      {
         assign_sd2sd(VOLUME,phi,psi);
         tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,status);
      }

      error_root(status[0]<0,1,"force1 [force1.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu,isp,status[0]);

      Dw_dble(-mu,chi,psi);
      mulg5_dble(VOLUME,psi);
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      if (sp.solver==SAP_GCR)
      {
         status[0]=0;
         status[1]=0;
      }
      else
      {
         for (l=0;l<6;l++)
            status[l]=0;
      }

      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(2);
         rho=rsd[0];
         eta=rsd[1];

         Dw_dble(-mu,chi,psi);
         mulg5_dble(VOLUME,psi);
         Dw_dble(mu,psi,rho);
         mulg5_dble(VOLUME,rho);
         mulr_spinor_add_dble(VOLUME,rho,phi,-1.0);

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME,1,phi);
            res1=unorm_dble(VOLUME,1,rho)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME,1,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME,1,rho);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               mulg5_dble(VOLUME,rho);

               if (sp.solver==SAP_GCR)
                  sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,rho,
                          eta,status);
               else
                  dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,rho,
                               eta,status);

               mulr_spinor_add_dble(VOLUME,psi,eta,-1.0);

               if (sp.istop)
               {
                  res0=unorm_dble(VOLUME,1,psi);
                  res1=unorm_dble(VOLUME,1,eta)/res0;
               }
               else
               {
                  rqsm=norm_square_dble(VOLUME,1,psi);
                  res0=rqsm.q[0];
                  rqsm=norm_square_dble(VOLUME,1,eta);
                  res1=sqrt(rqsm.q[0]/res0);
               }

               if (res1<1.0)
               {
                  if (res1>sp.res)
                  {
                     mulg5_dble(VOLUME,eta);

                     if (sp.solver==SAP_GCR)
                        sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res/res1,-mu,
                                eta,rho,status+1);
                     else
                        dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res/res1,-mu,
                                     eta,rho,status+3);

                     mulr_spinor_add_dble(VOLUME,chi,rho,-1.0);
                  }
               }
               else
               {
                  mulg5_dble(VOLUME,psi);

                  if (sp.solver==SAP_GCR)
                     sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,
                             chi,status+1);
                  else
                     dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,
                                  chi,status+3);

                  mulg5_dble(VOLUME,psi);
               }
            }
         }
         else
         {
            assign_sd2sd(VOLUME,phi,chi);
            mulg5_dble(VOLUME,chi);

            if (sp.solver==SAP_GCR)
               sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);
            else
               dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);

            mulg5_dble(VOLUME,psi);

            if (sp.solver==SAP_GCR)
               sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,status+1);
            else
               dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,status+3);

            mulg5_dble(VOLUME,psi);
         }

         release_wsd();
      }
      else
      {
         assign_sd2sd(VOLUME,phi,chi);
         mulg5_dble(VOLUME,chi);

         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);

         mulg5_dble(VOLUME,psi);

         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,status+1);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu,psi,chi,status+3);

         mulg5_dble(VOLUME,psi);
      }

      if (sp.solver==SAP_GCR)
         error_root((status[0]<0)||(status[1]<0),1,"force1 [force1.c]",
                    "SAP_GCR solver failed (mu = %.4e, parameter set no %d, "
                    "status = %d;%d)",mu,isp,status[0],status[1]);
      else
         error_root((status[0]<0)||(status[1]<0)||(status[3]<0)||(status[4]<0),
                    1,"force1 [force1.c]","DFL_SAP_GCR solver failed "
                    "(mu = %.4e, parameter set no %d, status = %d,%d,%d;"
                    "%d,%d,%d)",mu,isp,status[0],status[1],status[2],
                    status[3],status[4],status[5]);
   }
   else
      error(1,1,"force1 [force1.c]","Unknown solver");

   if (icr)
      add_chrono(icr,chi);

   set_xt2zero();
   add_prod2xt(1.0,chi,psi);
   sw_frc(c);

   set_xv2zero();
   add_prod2xv(1.0,chi,psi);
   hop_frc(c);

#if (defined FORCE_DBG)
   check_flds1(mu,sp,phi,psi,chi);
#endif

   release_wsd();
}


qflt action1(double mu,int ipf,int isp,int icr,int icom,int *status)
{
   int l;
   qflt rqsm,act;
   double res0,res1;
   spinor_dble *phi,*psi,*chi,**wsd;
   spinor_dble *rho,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg==1)
      set_tm_parms(0);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];
   chi=wsd[1];

   sw_term(NO_PTS);
   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(0,icr,mu,sp);
#endif

   if (sp.solver==CGNE)
   {
      status[0]=0;

      if (get_chrono(icr,chi))
      {
         rsd=reserve_wsd(1);
         rho=rsd[0];

         Dw_dble(-mu,chi,rho);
         mulg5_dble(VOLUME,rho);
         Dw_dble(mu,rho,psi);
         mulg5_dble(VOLUME,psi);
         mulr_spinor_add_dble(VOLUME,psi,phi,-1.0);

         release_wsd();

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME,1,phi);
            res1=unorm_dble(VOLUME,1,psi)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME,1,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME,1,psi);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               tmcg(sp.nmx,sp.istop,sp.res/res1,mu,psi,psi,status);
               mulr_spinor_add_dble(VOLUME,chi,psi,-1.0);
            }
         }
         else
         {
            assign_sd2sd(VOLUME,phi,psi);
            tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,status);
         }
      }
      else
      {
         assign_sd2sd(VOLUME,phi,psi);
         tmcg(sp.nmx,sp.istop,sp.res,mu,psi,chi,status);
      }

      error_root(status[0]<0,1,"action1 [force1.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",mu,isp,status[0]);

      Dw_dble(-mu,chi,psi);
      mulg5_dble(VOLUME,psi);
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      if (sp.solver==SAP_GCR)
         status[0]=0;
      else
      {
         for (l=0;l<3;l++)
            status[l]=0;
      }

      if (get_chrono(icr,chi))
      {
         Dw_dble(-mu,chi,psi);
         mulg5_dble(VOLUME,psi);
         Dw_dble(mu,psi,chi);
         mulg5_dble(VOLUME,chi);
         mulr_spinor_add_dble(VOLUME,chi,phi,-1.0);

         if (sp.istop)
         {
            res0=unorm_dble(VOLUME,1,phi);
            res1=unorm_dble(VOLUME,1,chi)/res0;
         }
         else
         {
            rqsm=norm_square_dble(VOLUME,1,phi);
            res0=rqsm.q[0];
            rqsm=norm_square_dble(VOLUME,1,chi);
            res1=sqrt(rqsm.q[0]/res0);
         }

         if (res1<1.0)
         {
            if (res1>sp.res)
            {
               mulg5_dble(VOLUME,chi);

               if (sp.solver==SAP_GCR)
                  sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,chi,
                          chi,status);
               else
                  dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res/res1,mu,chi,
                               chi,status);

               mulr_spinor_add_dble(VOLUME,psi,chi,-1.0);
            }
         }
         else
         {
            assign_sd2sd(VOLUME,phi,chi);
            mulg5_dble(VOLUME,chi);

            if (sp.solver==SAP_GCR)
               sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);
            else
               dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);
         }
      }
      else
      {
         assign_sd2sd(VOLUME,phi,chi);
         mulg5_dble(VOLUME,chi);

         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu,chi,psi,status);
      }

      if (sp.solver==SAP_GCR)
         error_root(status[0]<0,1,"action1 [force1.c]",
                    "SAP_GCR solver failed (mu = %.4e, parameter set no %d, "
                    "status = %d)",mu,isp,status[0]);
      else
         error_root((status[0]<0)||(status[1]<0),1,
                    "action1 [force1.c]","DFL_SAP_GCR solver failed "
                    "(mu = %.4e, parameter set no %d, status = %d,%d,%d)",
                    mu,isp,status[0],status[1],status[2]);
   }
   else
      error(1,1,"action1 [force1.c]","Unknown solver");

   act=norm_square_dble(VOLUME,icom,psi);

#if (defined FORCE_DBG)
   check_flds0(mu,sp,phi,psi);
#endif

   release_wsd();

   return act;
}
