
/*******************************************************************************
*
* File force5.c
*
* Copyright (C) 2011-2013, 2017, 2018 Stefan Schaefer, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Hasenbusch twisted mass pseudo-fermion action and force with even-odd
* preconditioning.
*
*   qflt setpf5(double mu0,double mu1,int ipf,int isp1,int icom,
*               int *status)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf-(phi,phi) (see the notes).
*
*   void rotpf5(double mu0,double mu1,int ipf,int isp0,int isp1,int icr,
*               double c1,double c2,int *status)
*     Generates a pseudo-fermion field eta with probability proportional
*     to exp(-Spf) and and replaces phi by c1*phi+c2*eta (see the notes).
*     The chronological solver stack with index icr is updated as well.
*
*   void force5(double mu0,int mu1,int ipf,int isp0,int icr,double c,
*               int *status)
*     Computes the force deriving from the action Spf (see the notes).
*     The calculated force is multiplied by c and added to the molecular-
*     dynamics force field.
*
*   qflt action5(double mu0,double mu1,int ipf,int isp0,int icr,
*                int icom,int *status)
*     Returns the action Spf-(phi,phi) (see the notes).
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,(Dwhat^dag*Dwhat+mu1^2)(Dwhat^dag*Dwhat+mu0^2)^(-1)*phi)
*
*      =(phi,phi)+(mu1^2-mu0^2)*(phi,(Dwhat^dag*Dwhat+mu0^2)^(-1)*phi)
*
* where Dwhat denotes the even-odd preconditioned (improved) Wilson-Dirac
* operator and phi the pseudo-fermion field. The latter vanishes on the
* odd lattice sites.
*
* The common parameters of the programs in this module are:
*
*   mu0,mu1       Twisted mass parameters in Spf.
*
*   ipf           Index of the pseudo-fermion field phi in the
*                 structure returned by mdflds() [mdflds.c].
*
*   isp0,isp1     Indices of the solver parameter sets that describes
*                 the solvers to be used for the solution of the Dirac
*                 equation with twisted mass mu0 and mu1.
*
*   icr           Index of the chronological solver stack to use.
*                 Setting icr=0 disables this feature.
*
*   icom          The action returned by the programs setpf5() and
*                 action5() is summed over all MPI processes if icom=1.
*                 Otherwise the local part of the action is returned.
*
*   status        Status values returned by the solver used for the
*                 solution of the Dirac equation.
*
* The supported solvers are CGNE, SAP_GCR and DFL_SAP_GCR. Depending
* on the program and the solver, the number of status variables is:
*
*                  CGNE         SAP_GCR       DFL_SAP_GCR
*   setpf5()         1             1               3
*   rotpf5()      1+(icr>0)     1+(icr>0)       3+3*(icr>0)
*   force5()         1             2               6
*   action5()        1             1               3
*
* The solver used in the case of setpf5() is for the Dirac equation with
* twisted mass mu1, while force5() and action5() use the solver for the
* equation with twisted mass mu0. Different solvers may be needed in the
* two cases if mu1>>mu0, for example.
*
* Note that, in force5(), the GCR solvers solve the Dirac equations twice.
* In these cases, the program writes the status values one after the other
* to the array. The bare quark mass m0 is the one last set by sw_parms()
* [flags/lat_parms.c] and it is taken for granted that the parameters of
* the solver have been set by set_solver_parms() [flags/solver_parms.c].
*
* The programs rotpf5() and force5() attempt to propagate the solutions of
* the Dirac equation along the molecular-dynamics trajectories, using the
* field stack number icr (no fields are propagated if icr=0). This feature
* assumes the program setup_chrono() [update/chrono.c] is called before
* rotpf5() or force5() is called for the first time.
*
* The required workspaces of double-precision spinor fields are
*
*                  CGNE         SAP_GCR       DFL_SAP_GCR
*   setpf5()         1             1               1
*   rotpf5()         2             2               2
*   force5()     2+(icr>0)     3+(icr>0)       3+(icr>0)
*   action5()        2             2               2
*
* (these figures do not include the workspace required by the solvers).
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define FORCE5_C

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

static void solver_info(int icr,double mu,solver_parms_t sp)
{
   if (sp.solver==CGNE)
      message("[rotpf5]: CGNE solver, icr = %d, istop = %d, mu = %.2e\n",
              icr,sp.istop,mu);
   else if (sp.solver==SAP_GCR)
      message("[rotpf5]: SAP_GCR solver, icr = %d, istop = %d, mu = %.2e\n",
              icr,sp.istop,mu);
   else if (sp.solver==DFL_SAP_GCR)
      message("[rotpf5]: DFL_SAP_GCR solver, icr = %d, istop = %d, mu = %.2e\n",
              icr,sp.istop,mu);
}


static void check_flds2(double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   sw_term(ODD_PTS);
   Dwhat_dble(-mu,psi,rho);
   mulg5_dble(VOLUME/2,rho);
   Dwhat_dble(mu,rho,psi);
   mulg5_dble(VOLUME/2,psi);
   mulr_spinor_add_dble(VOLUME/2,psi,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME/2,1,psi);
      rpsi/=unorm_dble(VOLUME/2,1,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME/2,1,psi);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME/2,1,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[rotpf5]: Residue of psi = %.1e\n",rpsi);

   release_wsd();
}

#endif

qflt setpf5(double mu0,double mu1,int ipf,int isp1,int icom,int *status)
{
   qflt act;
   complex_dble z;
   spinor_dble *phi,*psi,**wsd;
   spinor_dble *chi,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(1);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];

   random_sd(VOLUME/2,psi,1.0);
   bnd_sd2zero(EVEN_PTS,psi);
   assign_sd2sd(VOLUME/2,psi,phi);
   sp=solver_parms(isp1);

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.istop,sp.res,mu1,psi,psi,status);

      error_root(status[0]<0,1,"setpf5 [force5.c]","CGNE solver failed "
                 "(mu = %.4e, parameter set no %d, status = %d)",
                 mu1,isp1,status[0]);

      rsd=reserve_wsd(1);
      chi=rsd[0];
      assign_sd2sd(VOLUME/2,psi,chi);
      Dwhat_dble(-mu1,chi,psi);
      mulg5_dble(VOLUME/2,psi);
      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME/2,psi);
      set_sd2zero(VOLUME/2,psi+(VOLUME/2));

      if (sp.solver==SAP_GCR)
      {
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,
                 psi,psi,status);

         error_root(status[0]<0,1,"setpf5 [force5.c]","SAP_GCR solver failed "
                    "(mu = %.4e, parameter set no %d, status = %d)",
                    mu1,isp1,status[0]);
      }
      else
      {
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,
                      psi,psi,status);

         error_root((status[0]<0)||(status[1]<0),1,"setpf5 [force5.c]",
                    "DFL_SAP_GCR solver failed (mu = %.4e, parameter set "
                    "no %d, status = %d,%d,%d)",mu1,isp1,
                    status[0],status[1],status[2]);
      }
   }
   else
      error(1,1,"setpf5 [force5.c]","Unknown solver");

   z.re=0.0;
   z.im=mu0-mu1;
   mulc_spinor_add_dble(VOLUME/2,phi,psi,z);
   act=norm_square_dble(VOLUME/2,icom,psi);
   scl_qflt((mu1*mu1)-(mu0*mu0),act.q);

   release_wsd();

   return act;
}


void rotpf5(double mu0,double mu1,int ipf,int isp0,int isp1,int icr,
            double c1,double c2,int *status)
{
   complex_dble z;
   spinor_dble *phi,*psi,*eta,**wsd;
   spinor_dble *rho,**rsd;
   mdflds_t *mdfs;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(2);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   psi=wsd[0];
   eta=wsd[1];

   random_sd(VOLUME/2,eta,1.0);
   bnd_sd2zero(EVEN_PTS,eta);
   combine_spinor_dble(VOLUME/2,phi,eta,c1,c2);
   sp=solver_parms(isp1);

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.istop,sp.res,mu1,eta,psi,status);

      error_root(status[0]<0,1,"rotpf5 [force5.c]","CGNE solver failed "
                 "(mu = %.4e, parameter set no %d, status = %d)",
                 mu1,isp1,status[0]);

      Dwhat_dble(-mu1,psi,eta);
      mulg5_dble(VOLUME/2,eta);
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME/2,eta);
      set_sd2zero(VOLUME/2,eta+(VOLUME/2));

      if (sp.solver==SAP_GCR)
      {
         sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,
                 eta,eta,status);

         error_root(status[0]<0,1,"rotpf5 [force5.c]","SAP_GCR solver failed "
                    "(mu = %.4e, parameter set no %d, status = %d)",
                    mu1,isp1,status[0]);
      }
      else
      {
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu1,
                      eta,eta,status);

         error_root((status[0]<0)||(status[1]<0),1,"rotpf5 [force5.c]",
                    "DFL_SAP_GCR solver failed (mu = %.4e, parameter set "
                    "no %d, status = %d,%d,%d)",mu1,isp1,
                    status[0],status[1],status[2]);
      }
   }
   else
      error(1,1,"rotpf5 [force5.c]","Unknown solver");

   z.re=0.0;
   z.im=c2*(mu0-mu1);
   mulc_spinor_add_dble(VOLUME/2,phi,eta,z);

   if (get_chrono(icr,psi))
   {
      sp=solver_parms(isp0);

#if (defined FORCE_DBG)
      solver_info(icr,mu0,sp);
#endif

      if (sp.solver==CGNE)
      {
         tmcgeo(sp.nmx,sp.istop,sp.res,mu0,eta,eta,status+1);

         error_root(status[1]<0,1,"rotpf5 [force5.c]",
                    "CGNE solver failed (mu = %.4e, parameter set no %d, "
                    "status = %d)",mu0,isp0,status[1]);

         rsd=reserve_wsd(1);
         rho=rsd[0];
         Dwhat_dble(mu0,eta,rho);
         mulg5_dble(VOLUME/2,rho);
         combine_spinor_dble(VOLUME/2,psi,rho,c1,c2);
         release_wsd();
      }
      else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
      {
         sap=sap_parms();
         set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
         mulg5_dble(VOLUME/2,eta);
         set_sd2zero(VOLUME/2,eta+(VOLUME/2));

         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu0,eta,
                    eta,status+1);

            error_root(status[1]<0,1,"rotpf5 [force5.c]",
                       "SAP_GCR solver failed (mu = %.4e, "
                       "parameter set no %d, status = %d)",mu0,isp0,status[1]);
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu0,eta,
                         eta,status+3);

            error_root((status[3]<0)||(status[4]<0),1,"rotpf5 [force5.c]",
                       "DFL_SAP_GCR solver failed (mu = %.4e, "
                       "parameter set no %d, status = %d,%d,%d",
                       mu0,isp0,status[3],status[4],status[5]);
         }

         combine_spinor_dble(VOLUME/2,psi,eta,c1,c2);
      }
      else
         error(1,1,"rotpf5 [force5.c]","Unknown solver");

      add_chrono(icr,psi);

#if (defined FORCE_DBG)
      check_flds2(mu0,sp,phi,psi);
#endif
   }
   else if (icr>0)
   {
      sp=solver_parms(isp0);

      if (sp.solver==DFL_SAP_GCR)
      {
         status[3]=0;
         status[4]=0;
         status[5]=0;
      }
      else
         status[1]=0;
   }

   release_wsd();
}


void force5(double mu0,double mu1,int ipf,int isp,int icr,
            double c,int *status)
{
   double dmu2;

   dmu2=(mu1*mu1)-(mu0*mu0);
   force4(mu0,ipf,0,isp,icr,dmu2*c,status);
}


qflt action5(double mu0,double mu1,int ipf,int isp,int icr,
             int icom,int *status)
{
   qflt act;

   act=action4(mu0,ipf,0,isp,icr,icom,status);
   scl_qflt((mu1*mu1)-(mu0*mu0),act.q);

   return act;
}
