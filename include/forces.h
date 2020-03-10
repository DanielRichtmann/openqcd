
/*******************************************************************************
*
* File forces.h
*
* Copyright (C) 2011, 2012, 2018 Martin Luescher, Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef FORCES_H
#define FORCES_H

#ifndef SU3_H
#include "su3.h"
#endif

#ifndef UTILS_H
#include "utils.h"
#endif

/* FORCE0_C */
extern void plaq_frc(void);
extern void force0(double c);
extern qflt action0(int icom);

/* FORCE1_C */
extern qflt setpf1(double mu,int ipf,int icom);
extern void rotpf1(double mu,int ipf,int isp,int icr,
                   double c1,double c2,int *status);
extern void force1(double mu,int ipf,int isp,int icr,
                   double c,int *status);
extern qflt action1(double mu,int ipf,int isp,int icr,
                    int icom,int *status);

/* FORCE2_C */
extern qflt setpf2(double mu0,double mu1,int ipf,int isp1,
                   int icom,int *status);
extern void rotpf2(double mu0,double mu1,int ipf,int isp0,int isp1,int icr,
                   double c1,double c2,int *status);
extern void force2(double mu0,double mu1,int ipf,int isp0,int icr,
                   double c,int *status);
extern qflt action2(double mu0,double mu1,int ipf,int isp0,int icr,
                    int icom,int *status);

/* FORCE3_C */
extern qflt setpf3(int *irat,int ipf,int isw,int isp,int icom,int *status);
extern void rotpf3(int *irat,int ipf,int isp,double c1,double c2,int *status);
extern void force3(int *irat,int ipf,int isw,int isp,double c,int *status);
extern qflt action3(int *irat,int ipf,int isw,int isp,int icom,int *status);

/* FORCE4_C */
extern qflt setpf4(double mu,int ipf,int isw,int icom);
extern void rotpf4(double mu,int ipf,int isp,int icr,
                   double c1,double c2,int *status);
extern void force4(double mu,int ipf,int isw,int isp,int icr,
                   double c,int *status);
extern qflt action4(double mu,int ipf,int isw,int isp,int icr,
                    int icom,int *status);

/* FORCE5_C */
extern qflt setpf5(double mu0,double mu1,int ipf,int isp1,int icom,
                   int *status);
void rotpf5(double mu0,double mu1,int ipf,int isp0,int isp1,int icr,
            double c1,double c2,int *status);
extern void force5(double mu0,double mu1,int ipf,int isp0,int icr,
                   double c,int *status);
extern qflt action5(double mu0,double mu1,int ipf,int isp0,int icr,
                    int icom,int *status);

/* FRCFCTS_C */
extern void cmb6weyl(double *c,weyl_dble *s,weyl_dble *r);
extern void det2xt(pauli_dble *m,u3_alg_dble *X);
extern void prod2xt(spinor_dble *r,spinor_dble *s,u3_alg_dble *X);
extern void (*prod2xv[])(spinor_dble *rx,spinor_dble *ry,
                         spinor_dble *sx,spinor_dble *sy,su3_dble *u);

/* GENFRC_C */
extern void sw_frc(double c);
extern void hop_frc(double c);

/* TMCG_C */
extern double tmcg(int nmx,int istop,double res,double mu,
                   spinor_dble *eta,spinor_dble *psi,int *status);
extern double tmcgeo(int nmx,int istop,double res,double mu,
                     spinor_dble *eta,spinor_dble *psi,int *status);

/* TMCGM_C */
extern void tmcgm(int nmx,int istop,double *res,int nmu,double *mu,
                  spinor_dble *eta,spinor_dble **psi,int *status);

/* XTENSOR_C */
extern u3_alg_dble **xtensor(void);
extern void set_xt2zero(void);
extern int add_det2xt(double c,ptset_t set);
extern void add_prod2xt(double c,spinor_dble *r,spinor_dble *s);
extern su3_dble *xvector(void);
extern void set_xv2zero(void);
extern void add_prod2xv(double c,spinor_dble *r,spinor_dble *s);

#endif
