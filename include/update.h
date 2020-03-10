
/*******************************************************************************
*
* File update.h
*
* Copyright (C) 2011, 2017, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef UPDATE_H
#define UPDATE_H

#ifndef SU3_H
#include "su3.h"
#endif

typedef struct
{
   int iop;
   double eps;
} mdstep_t;

/* CHRONO_C */
extern void setup_chrono(void);
extern double mdtime(void);
extern void step_mdtime(double dt);
extern void add_chrono(int icr,spinor_dble *psi);
extern int get_chrono(int icr,spinor_dble *psi);
extern void reset_chrono(int n);
extern size_t chrono_msize(void);

/* COUNTERS_C */
extern void setup_counters(void);
extern void clear_counters(void);
extern void add2counter(char *type,int idx,int *status);
extern int get_count(char *type,int idx,int *status);
extern void print_avgstat(char *type,int idx);
extern void print_all_avgstat(void);

/* HMC_C */
extern int run_hmc(qflt *act0,qflt *act1);

/* MDINT_C */
extern void run_mdint(void);

/* MDSTEPS_C */
extern void set_mdsteps(void);
extern mdstep_t *mdsteps(int *nop,int *itu);
extern void print_mdsteps(int ipr);

/* MSIZE_C */
extern size_t msize(void);
extern void print_msize(int ipr);

/* RWRAT_C */
extern qflt rwrat(int irp,int n,int *np,int *isp,qflt *sqn,int **status);

/* RWTM_C */
extern qflt rwtm1(double mu1,double mu2,int isp,qflt *sqn,int *status);
extern qflt rwtm2(double mu1,double mu2,int isp,qflt *sqn,int *status);

/* RWTMEO_C */
extern qflt rwtm1eo(double mu1,double mu2,int isp,qflt *sqn,int *status);
extern qflt rwtm2eo(double mu1,double mu2,int isp,qflt *sqn,int *status);

/* SANITY_C */
extern void hmc_sanity_check(void);
extern void smd_sanity_check(void);
extern int matching_force(int iact);

/* SMD_C */
extern void smd_reset_dfl(void);
extern void smd_init(void);
extern void smd_action(qflt *act);
extern int run_smd(qflt *act0,qflt *act1);
extern void run_smd_noacc0(qflt *act0,qflt *act1);
extern void run_smd_noacc1(void);

/* UPDATE_C */
extern void update_mom(void);
extern void update_ud(double eps);
extern void start_dfl_upd(void);
extern void dfl_upd(void);

/* WSIZE_C */
extern void hmc_wsize(int *nwud,int *nws,int *nwv,int *nwvd);
extern void smd_wsize(int *nwud,int *nwfd,int *nws,int *nwv,int *nwvd);

#endif
