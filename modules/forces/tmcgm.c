
/*******************************************************************************
*
* File tmcgm.c
*
* Copyright (C) 2012, 2013, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Multi-shift CG solver for the normal even-odd preconditioned Wilson-Dirac
* equation (Dwhat^dag*Dwhat+mu^2)*psi=eta with a twisted-mass term.
*
*   void tmcgm(int nmx,int istop,double *res,int nmu,double *mu,
*              spinor_dble *eta,spinor_dble **psi,int *status)
*     Obtains approximate solutions psi[0],..,psi[nmu-1] of the normal
*     even-odd preconditioned Wilson-Dirac equation for given source eta
*     and nmu values of the twisted-mass parameter mu.
*
* The program is based on the multi-shift CG algorithm (see linsolv/mscg.c).
* It assumes that the improvement coefficients and the quark mass in the
* SW term have been set through set_lat_parms() and set_sw_parms() (see
* flags/lat_parms.c).
*
* All other parameters are passed through the argument list:
*
*   nmx     Maximal total number of CG iterations that may be performed.
*
*   istop   Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*   res     Array of the desired maximal relative residues of the
*           calculated solutions (nmu elements)
*
*   nmu     Number of twisted masses mu.
*
*   mu      Array of the twisted masses (nmu elements)
*
*   eta     Source field. On exit eta is unchanged.
*
*   psi     Array of the calculated approximate solutions of the Dirac
*           equations (Dwhat^dag*Dwhat+mu^2)*psi=eta (nmu elements).
*           The fields psi[0],..,psi[nmu-1] must be different from eta.
*
*   status  If the program was able to solve the Dirac equations to the
*           desired accuracy, status[0] reports the total number of CG
*           iterations that were required. Negative values indicate that
*           the program failed (-1: the algorithm did not converge, -2:
*           the inversion of the SW term on the odd points was not safe).
*
* The fields eta and psi must be such that the Dirac operator can act on
* them (see main/README.global). Moreover, the source eta is assumed to
* respect the chosen boundary conditions (see doc/dirac.pdf).
*
* If status[0]>=-1 the calculated approximate solutions are returned. The
* fields are otherwise set to zero.
*
* The SW term is recalculated when needed. The solver is a global program that
* must be called on all processes simultaneously. The required workspace is
*
*  spinor_dble         3+nmu (5 if nmu=1)
*
* (see utils/wspace.c).
*
*******************************************************************************/

#define TMCGM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "linsolv.h"
#include "forces.h"
#include "global.h"

static int iop=0;


static void Dop_dble(double mu,spinor_dble *s,spinor_dble *r)
{
   if (iop==0)
      Dwhat_dble(mu,s,r);
   else
      Dwhat_dble(-mu,s,r);

   mulg5_dble(VOLUME/2,r);
   iop^=0x1;
}


void tmcgm(int nmx,int istop,double *res,int nmu,double *mu,
           spinor_dble *eta,spinor_dble **psi,int *status)
{
   int ifail,k;
   double rho0;
   spinor_dble **wsd;

   ifail=sw_term(ODD_PTS);

   if (ifail)
   {
      status[0]=-2;

      for (k=0;k<nmu;k++)
         set_sd2zero(VOLUME/2,psi[k]);
   }
   else
   {
      rho0=unorm_dble(VOLUME/2,1,eta);

      if (rho0!=0.0)
      {
         if (nmu==1)
            wsd=reserve_wsd(5);
         else
            wsd=reserve_wsd(3+nmu);

         assign_sd2sd(VOLUME/2,eta,wsd[0]+(VOLUME/2));
         scale_dble(VOLUME/2,1.0/rho0,eta);

         mscg(VOLUME/2,1,nmu,mu,Dop_dble,wsd,nmx,istop,res,eta,psi,status);

         for (k=0;k<nmu;k++)
            scale_dble(VOLUME/2,rho0,psi[k]);

         assign_sd2sd(VOLUME/2,wsd[0]+(VOLUME/2),eta);
         release_wsd();
      }
      else
      {
         status[0]=0;

         for (k=0;k<nmu;k++)
            set_sd2zero(VOLUME/2,psi[k]);
      }
   }
}
