
/*******************************************************************************
*
* File auxfcts.h
*
* Copyright (C) 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef AUXFCTS_H
#define AUXFCTS_H

extern void save_ranlux(void);
extern void restore_ranlux(void);
extern void rot_ud(double eps);
extern void check_bnd_fld(su3_alg_dble *fld);

#endif
