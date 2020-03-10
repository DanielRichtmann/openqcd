
/*******************************************************************************
*
* File cmpflds.h
*
* Copyright (C) 2017 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef CMPFLDS_H
#define CMPFLDS_H

#ifndef SU3_H
#include "su3.h"
#endif

/* CMPFLDS */
extern int check_ud(int vol,su3_dble *ud,su3_dble *vd);
extern int check_fd(int vol,su3_alg_dble *fd,su3_alg_dble *gd);
extern int check_sd(int vol,spinor_dble *sd,spinor_dble *rd);

#endif
