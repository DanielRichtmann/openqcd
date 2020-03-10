
/*******************************************************************************
*
* File tmcg.c
*
* Copyright (C) 2011-2013, 2018 Martin Luescher, Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* CG solver for the normal Wilson-Dirac equation with a twisted-mass term.
*
*   double tmcg(int nmx,int istop,double res,double mu,
*               spinor_dble *eta,spinor_dble *psi,int *status)
*     Obtains an approximate solution psi of the normal Wilson-Dirac
*     equation for given source eta.
*
*   double tmcgeo(int nmx,int istop,double res,double mu,
*                 spinor_dble *eta,spinor_dble *psi,int *status)
*     Obtains an approximate solution psi of the normal even-odd
*     preconditioned Wilson-Dirac equation for given source eta. On
*     the odd lattice points, the fields eta and psi are unchanged.
*
* The normal and the normal even-odd preconditioned Wilson-Dirac equations
* are
*
*   (Dw^dag*Dw+mu^2)*psi=eta,
*
*   (Dwhat^dag*Dwhat+mu^2)*psi=eta,
*
* respectively.
*
* The programs are based on the standard CG algorithm (see linsolv/cgne.c).
* They assume that the improvement coefficients and the quark mass in the
* SW term have been set through set_lat_parms() and set_sw_parms() (see
* flags/lat_parms.c).
*
* All other parameters are passed through the argument list:
*
*   nmx     Maximal total number of CG iterations that may be performed.
*
*   istop   Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*   res     Desired maximal relative residue of the calculated solution.
*
*   mu      Value of the twisted mass in the Dirac equation.
*
*   eta     Source field. On exit eta is unchanged unless psi=eta (which
*           is permissible).
*
*   psi     Calculated approximate solution of the Dirac equation.
*
*   status  If the program terminates normally, status reports the total
*           number of CG iterations that were required for the solution of
*           the Dirac equation. Status is set to -1 if the solver did not
*           converge and to -2 if [in the case of tmcgeo()] the inversion
*           of the SW term was not safe.
*
* The fields eta and psi must be such that the Dirac operator can act on
* them (see main/README.global). Moreover, the source eta is assumed to
* respect the chosen boundary conditions (see doc/dirac.pdf).

* The programs return the norm of the residue of the calculated approximate
* solution if status[0]>=-1. Otherwise the program returns the norm of the
* source eta and sets psi to zero if psi!=eta.
*
* The SW term is recalculated when needed. The solver is a global program that
* must be called on all processes simultaneously. The required workspaces are
*
*  spinor              5
*  spinor_dble         3
*
* (see utils/wspace.c).
*
*******************************************************************************/

#define TMCG_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "flags.h"
#include "utils.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "linsolv.h"
#include "forces.h"
#include "global.h"

static float mus;
static double mud;


static void Dop(spinor *s,spinor *r)
{
   Dw(mus,s,r);
   mulg5(VOLUME,r);
   mus=-mus;
}


static void Dop_dble(spinor_dble *s,spinor_dble *r)
{
   Dw_dble(mud,s,r);
   mulg5_dble(VOLUME,r);
   mud=-mud;
}


double tmcg(int nmx,int istop,double res,double mu,
            spinor_dble *eta,spinor_dble *psi,int *status)
{
   int eoflg;
   double rho0,rho;
   spinor **ws;
   spinor_dble **wsd,*rsd;
   tm_parms_t tm;

   tm=tm_parms();
   eoflg=tm.eoflg;
   if (eoflg)
      set_tm_parms(0);

   if (query_flags(U_MATCH_UD)!=1)
      assign_ud2u();

   sw_term(NO_PTS);

   if ((query_flags(SW_UP2DATE)!=1)||
       (query_flags(SW_E_INVERTED)!=0)||(query_flags(SW_O_INVERTED)!=0))
      assign_swd2sw();

   status[0]=0;
   rho=0.0;
   rho0=unorm_dble(VOLUME,1,eta);

   if (rho0!=0.0)
   {
      ws=reserve_ws(5);
      wsd=reserve_wsd(3);
      rsd=wsd[2];

      mus=(float)(mu);
      mud=mu;
      assign_sd2sd(VOLUME,eta,rsd);
      scale_dble(VOLUME,1.0/rho0,rsd);

      rho=cgne(VOLUME,1,Dop,Dop_dble,ws,wsd,nmx,istop,res,rsd,
               psi,status);

      scale_dble(VOLUME,rho0,psi);
      rho*=rho0;

      release_wsd();
      release_ws();
   }
   else if (psi!=eta)
      set_sd2zero(VOLUME,psi);

   if (eoflg)
      set_tm_parms(1);

   return rho;
}


static void Doph(spinor *s,spinor *r)
{
   Dwhat(mus,s,r);
   mulg5(VOLUME/2,r);
   mus=-mus;
}


static void Doph_dble(spinor_dble *s,spinor_dble *r)
{
   Dwhat_dble(mud,s,r);
   mulg5_dble(VOLUME/2,r);
   mud=-mud;
}


double tmcgeo(int nmx,int istop,double res,double mu,
              spinor_dble *eta,spinor_dble *psi,int *status)
{
   int ifail;
   double rho0,rho;
   qflt rqsm;
   spinor **ws;
   spinor_dble **wsd,*rsd;

   ifail=sw_term(ODD_PTS);
   status[0]=0;
   rho=0.0;

   if (ifail)
      status[0]=-2;
   else
   {
      if (query_flags(U_MATCH_UD)!=1)
         assign_ud2u();

      if ((query_flags(SW_UP2DATE)!=1)||
          (query_flags(SW_E_INVERTED)!=0)||(query_flags(SW_O_INVERTED)!=1))
         assign_swd2sw();

      rho0=unorm_dble(VOLUME/2,1,eta);

      if (rho0!=0.0)
      {
         ws=reserve_ws(5);
         wsd=reserve_wsd(3);
         rsd=wsd[2];

         mus=(float)(mu);
         mud=mu;
         assign_sd2sd(VOLUME/2,eta,rsd);
         scale_dble(VOLUME/2,1.0/rho0,rsd);

         rho=cgne(VOLUME/2,1,Doph,Doph_dble,ws,wsd,nmx,istop,res,rsd,
                  psi,status);

         scale_dble(VOLUME/2,rho0,psi);
         rho*=rho0;

         release_wsd();
         release_ws();
      }
      else if (psi!=eta)
         set_sd2zero(VOLUME/2,psi);
   }

   if (status[0]<-1)
   {
      if (psi!=eta)
         set_sd2zero(VOLUME/2,psi);

      if (istop)
         rho=unorm_dble(VOLUME/2,1,eta);
      else
      {
         rqsm=norm_square_dble(VOLUME/2,1,eta);
         rho=sqrt(rqsm.q[0]);
      }
   }

   return rho;
}
