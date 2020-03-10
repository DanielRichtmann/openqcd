
/*******************************************************************************
*
* File ym2.c
*
* Copyright (C) 2017-2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* SMD simulation program for the SU(3) gauge theory.
*
* Syntax: ym2 -i <filename> [-c <filename> [-a [-norng]]|[-mask <int>]]
*                           [-rmmom] [-rmold] [-noms]
*
* For usage instructions see the file README.ym2.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "archive.h"
#include "forces.h"
#include "update.h"
#include "wflow.h"
#include "tcharge.h"
#include "version.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

typedef struct
{
   int nc,iac;
   double dH,avpl;
} dat_t;

static struct
{
   int dn,nn,tmax;
   double eps;
} file_head;

static struct
{
   int nc;
   double **Wsl,**Ysl,**Qsl;
} data;

static struct
{
   int type,nio_nodes,nio_streams;
   int nb,ib,bs[4];
} iodat[2];

static int my_rank,mask,norng,rmmom,rmold,noms;
static int scnfg,append,endian,level,seed;
static int nth,ntot,dnlog,dncnfg,dnms;
static int ipgrd[2],flint;
static double *Wact,*Yact,*Qtop;

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char cnfg_dir[NAME_SIZE],blk_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],init_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char msdat_file[NAME_SIZE],msdat_save[NAME_SIZE];
static char rng_file[NAME_SIZE],rng_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],end_file[NAME_SIZE];
static char nbase[NAME_SIZE],cnfg[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;

static lat_parms_t lat;
static smd_parms_t smd;


static int write_dat(int n,dat_t *ndat)
{
   int i,iw,ic;
   stdint_t istd[2];
   double dstd[2];

   ic=0;

   for (i=0;i<n;i++)
   {
      istd[0]=(stdint_t)((*ndat).nc);
      istd[1]=(stdint_t)((*ndat).iac);

      dstd[0]=(*ndat).dH;
      dstd[1]=(*ndat).avpl;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(2,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),2,fdat);
      iw+=fwrite(dstd,sizeof(double),2,fdat);

      if (iw!=4)
         return ic;

      ic+=1;
      ndat+=1;
   }

   return ic;
}


static int read_dat(int n,dat_t *ndat)
{
   int i,ir,ic;
   stdint_t istd[2];
   double dstd[2];

   ic=0;

   for (i=0;i<n;i++)
   {
      ir=fread(istd,sizeof(stdint_t),2,fdat);
      ir+=fread(dstd,sizeof(double),2,fdat);

      if (ir!=4)
         return ic;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(2,dstd);
      }

      (*ndat).nc=(int)(istd[0]);
      (*ndat).iac=(int)(istd[1]);

      (*ndat).dH=dstd[0];
      (*ndat).avpl=dstd[1];

      ic+=1;
      ndat+=1;
   }

   return ic;
}


static void alloc_data(void)
{
   int nn,tmax;
   int in,k;
   double **pp,*p;

   nn=file_head.nn;
   tmax=file_head.tmax;

   pp=amalloc(3*(nn+1)*sizeof(*pp),3);
   p=amalloc(3*(nn+1)*(tmax+1)*sizeof(*p),4);
   error((pp==NULL)||(p==NULL),1,"alloc_data [ym2.c]",
         "Unable to allocate data arrays");
   for (k=0;k<(3*(nn+1)*(tmax+1));k++)
      p[k]=0.0;

   data.Wsl=pp;
   data.Ysl=pp+nn+1;
   data.Qsl=pp+2*(nn+1);

   for (in=0;in<(3*(nn+1));in++)
   {
      *pp=p;
      pp+=1;
      p+=tmax;
   }

   Wact=p;
   p+=nn+1;
   Yact=p;
   p+=nn+1;
   Qtop=p;
}


static void write_file_head(void)
{
   int iw;
   stdint_t istd[3];
   double dstd[1];

   istd[0]=(stdint_t)(file_head.dn);
   istd[1]=(stdint_t)(file_head.nn);
   istd[2]=(stdint_t)(file_head.tmax);
   dstd[0]=file_head.eps;

   if (endian==BIG_ENDIAN)
   {
      bswap_int(3,istd);
      bswap_double(1,dstd);
   }

   iw=fwrite(istd,sizeof(stdint_t),3,fdat);
   iw+=fwrite(dstd,sizeof(double),1,fdat);

   error_root(iw!=4,1,"write_file_head [ym2.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int ir;
   stdint_t istd[3];
   double dstd[1];

   ir=fread(istd,sizeof(stdint_t),3,fdat);
   ir+=fread(dstd,sizeof(double),1,fdat);

   error_root(ir!=4,1,"check_file_head [ym2.c]",
              "Incorrect read count");

   if (endian==BIG_ENDIAN)
   {
      bswap_int(3,istd);
      bswap_double(1,dstd);
   }

   error_root(((int)(istd[0])!=file_head.dn)||
              ((int)(istd[1])!=file_head.nn)||
              ((int)(istd[2])!=file_head.tmax)||
              (dstd[0]!=file_head.eps),1,"check_file_head [ym2.c]",
              "Unexpected value of dn,nn,tmax or eps");
}


static void write_data(void)
{
   int iw,nn,tmax;
   int in,t;
   stdint_t istd[1];
   double dstd[1];

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   nn=file_head.nn;
   tmax=file_head.tmax;

   for (in=0;in<=nn;in++)
   {
      for (t=0;t<tmax;t++)
      {
         dstd[0]=data.Wsl[in][t];

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         iw+=fwrite(dstd,sizeof(double),1,fdat);
      }
   }

   for (in=0;in<=nn;in++)
   {
      for (t=0;t<tmax;t++)
      {
         dstd[0]=data.Ysl[in][t];

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         iw+=fwrite(dstd,sizeof(double),1,fdat);
      }
   }

   for (in=0;in<=nn;in++)
   {
      for (t=0;t<tmax;t++)
      {
         dstd[0]=data.Qsl[in][t];

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         iw+=fwrite(dstd,sizeof(double),1,fdat);
      }
   }

   error_root(iw!=(1+3*(nn+1)*tmax),1,"write_data [ym2.c]",
              "Incorrect write count");
}


static int read_data(void)
{
   int ir,nn,tmax;
   int in,t;
   stdint_t istd[1];
   double dstd[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   data.nc=(int)(istd[0]);

   nn=file_head.nn;
   tmax=file_head.tmax;

   for (in=0;in<=nn;in++)
   {
      for (t=0;t<tmax;t++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         data.Wsl[in][t]=dstd[0];
      }
   }

   for (in=0;in<=nn;in++)
   {
      for (t=0;t<tmax;t++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         data.Ysl[in][t]=dstd[0];
      }
   }

   for (in=0;in<=nn;in++)
   {
      for (t=0;t<tmax;t++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);

         data.Qsl[in][t]=dstd[0];
      }
   }

   error_root(ir!=(1+3*(nn+1)*tmax),1,"read_data [ym2.c]",
              "Read error or incomplete data record");

   return 1;
}


static void read_dirs(void)
{
   int type,nion,nios,bs[4];

   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Log directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      if (scnfg)
      {
         find_section("Initial configuration");
         read_line("type","%s",line);

         if (strchr(line,'e')!=NULL)
            type=0x1;
         else if (strchr(line,'b')!=NULL)
            type=0x2;
         else if (strchr(line,'l')!=NULL)
            type=0x4;
         else
            type=0x0;

         error_root(type==0x0,1,"read_dirs [ym2.c]",
                    "No valid initial-configuration storage type specified");

         read_line("init_dir","%s",init_dir);

         if (type&0x6)
         {
            read_line("nio_nodes","%d",&nion);
            read_line("nio_streams","%d",&nios);
         }
         else
         {
            nion=1;
            nios=0;
         }
      }
      else
      {
         type=0x0;
         init_dir[0]='\0';
         nion=1;
         nios=0;
      }
   }

   MPI_Bcast(init_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);

   iodat[0].type=type;
   iodat[0].nio_nodes=nion;
   iodat[0].nio_streams=nios;
   iodat[0].nb=0;
   iodat[0].ib=NPROC;
   iodat[0].bs[0]=0;
   iodat[0].bs[1]=0;
   iodat[0].bs[2]=0;
   iodat[0].bs[3]=0;

   if (my_rank==0)
   {
      find_section("Configurations");
      read_line("types","%s",line);

      type=0x0;
      if (strchr(line,'e')!=NULL)
         type|=0x1;
      if (strchr(line,'b')!=NULL)
         type|=0x2;
      if (strchr(line,'l')!=NULL)
         type|=0x4;

      error_root(type==0x0,1,"read_dirs [ym2.c]",
                 "No valid configuration storage type specified");

      if (type&0x1)
         read_line("cnfg_dir","%s",cnfg_dir);
      else
         cnfg_dir[0]='\0';

      if (type&0x2)
         read_line("block_dir","%s",blk_dir);
      else
         blk_dir[0]='\0';

      if (type&0x4)
         read_line("local_dir","%s",loc_dir);
      else
         loc_dir[0]='\0';

      if (type&0x2)
         read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      else
      {
         bs[0]=0;
         bs[1]=0;
         bs[2]=0;
         bs[3]=0;
      }

      if (type&0x6)
      {
         read_line("nio_nodes","%d",&nion);
         read_line("nio_streams","%d",&nios);
      }
      else
      {
         nion=1;
         nios=0;
      }
   }

   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(blk_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&type,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nion,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nios,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);

   iodat[1].type=type;
   iodat[1].nio_nodes=nion;
   iodat[1].nio_streams=nios;
   iodat[1].nb=0;
   iodat[1].ib=NPROC;
   iodat[1].bs[0]=bs[0];
   iodat[1].bs[1]=bs[1];
   iodat[1].bs[2]=bs[2];
   iodat[1].bs[3]=bs[3];
}


static void setup_files(void)
{
   check_dir_root(log_dir);
   check_dir_root(dat_dir);

   error(name_size("%s/%s.log~",log_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ym2.c]","log_dir name is too long");
   error(name_size("%s/%s.ms.dat~",dat_dir,nbase)>=NAME_SIZE,1,
         "setup_files [ym2.c]","dat_dir name is too long");

   sprintf(log_file,"%s/%s.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.par",dat_dir,nbase);
   sprintf(dat_file,"%s/%s.dat",dat_dir,nbase);
   sprintf(msdat_file,"%s/%s.ms.dat",dat_dir,nbase);
   sprintf(rng_file,"%s/%s.rng",dat_dir,nbase);
   sprintf(end_file,"%s/%s.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);
   sprintf(dat_save,"%s~",dat_file);
   sprintf(msdat_save,"%s~",msdat_file);
   sprintf(rng_save,"%s~",rng_file);
}


static void read_lat_parms(void)
{
   double beta,c0;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("beta","%lf",&beta);
      read_line("c0","%lf",&c0);
   }

   MPI_Bcast(&beta,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&c0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   lat=set_lat_parms(beta,c0,0,NULL,0,1.0);

   if (append)
      check_lat_parms(fdat);
   else
      write_lat_parms(fdat);
}


static void read_bc_parms(void)
{
   int bc;
   double cG,cG_prime;
   double phi[2],phi_prime[2],theta[3];

   if (my_rank==0)
   {
      find_section("Boundary conditions");
      read_line("type","%d",&bc);

      phi[0]=0.0;
      phi[1]=0.0;
      phi_prime[0]=0.0;
      phi_prime[1]=0.0;
      cG=1.0;
      cG_prime=1.0;

      if (bc==1)
         read_dprms("phi",2,phi);

      if ((bc==1)||(bc==2))
         read_dprms("phi'",2,phi_prime);

      if (bc!=3)
         read_line("cG","%lf",&cG);

      if (bc==2)
         read_line("cG'","%lf",&cG_prime);
   }

   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(phi,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(phi_prime,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cG_prime,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   theta[0]=0.0;
   theta[1]=0.0;
   theta[2]=0.0;

   set_bc_parms(bc,cG,cG_prime,1.0,1.0,phi,phi_prime,theta);

   if (append)
      check_bc_parms(fdat);
   else
      write_bc_parms(fdat);
}


static void read_smd_parms(void)
{
   int iacc,iact[1];
   double gamma,eps;

   if (my_rank==0)
   {
      find_section("SMD parameters");
      read_line("gamma","%lf",&gamma);
      read_line("eps","%lf",&eps);
      read_line("iacc","%d",&iacc);
   }

   MPI_Bcast(&gamma,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&iacc,1,MPI_INT,0,MPI_COMM_WORLD);

   iact[0]=0;
   smd=set_smd_parms(1,iact,0,0,NULL,1,gamma,eps,iacc);

   if (append)
      check_smd_parms(fdat);
   else
      write_smd_parms(fdat);
}


static void read_integrator(void)
{
   int imd,nstep,ifr[1];
   double lambda;

   if (my_rank==0)
   {
      find_section("MD integrator");
      read_line("integrator","%s",line);
      lambda=0.0;

      if (strcmp(line,"LPFR")==0)
         imd=(int)(LPFR);
      else if (strcmp(line,"OMF2")==0)
      {
         imd=(int)(OMF2);
         read_line("lambda","%lf",&lambda);
      }
      else if (strcmp(line,"OMF4")==0)
         imd=(int)(OMF4);
      else
         error_root(1,1,"read_integrator [ym2.c]","Unknown integrator");

      read_line("nstep","%d",&nstep);
   }

   MPI_Bcast(&imd,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&lambda,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nstep,1,MPI_INT,0,MPI_COMM_WORLD);

   ifr[0]=0;
   set_mdint_parms(0,(integrator_t)(imd),lambda,nstep,1,ifr);
   set_action_parms(0,ACG,0,0,NULL,NULL,NULL);
   set_force_parms(0,FRG,0,0,NULL,NULL,NULL,NULL);

   if (append)
   {
      check_mdint_parms(fdat);
      check_action_parms(fdat);
      check_force_parms(fdat);
   }
   else
   {
      write_mdint_parms(fdat);
      write_action_parms(fdat);
      write_force_parms(fdat);
   }
}


static void read_wflow_parms(void)
{
   int nt,dn,ie,ir,iw;
   stdint_t istd[3];
   double eps,dstd[1];

   if (my_rank==0)
   {
      if (append)
      {
         ir=fread(istd,sizeof(stdint_t),1,fdat);
         error_root(ir!=1,1,"read_wflow_parms [ym2.c]",
                    "Incorrect read count");

         if (endian==BIG_ENDIAN)
            bswap_int(1,istd);

         error_root(istd[0]!=(stdint_t)(noms==0),1,"read_wflow_parms [ym2.c]",
                    "Attempt to mix measurement with other runs");
      }
      else
      {
         istd[0]=(stdint_t)(noms==0);

         if (endian==BIG_ENDIAN)
            bswap_int(1,istd);

         iw=fwrite(istd,sizeof(stdint_t),1,fdat);
         error_root(iw!=1,1,"read_wflow_parms [ym2.c]",
                    "Incorrect write count");
      }

      if (noms==0)
      {
         find_section("Wilson flow");
         read_line("integrator","%s",line);
         read_line("eps","%lf",&eps);
         read_line("ntot","%d",&nt);
         read_line("dnms","%d",&dn);

         if (strcmp(line,"EULER")==0)
            flint=0;
         else if (strcmp(line,"RK2")==0)
            flint=1;
         else if (strcmp(line,"RK3")==0)
            flint=2;
         else
            error_root(1,1,"read_wflow_parms [ym2.c]","Unkown integrator");

         error_root((dn<1)||(nt<dn)||((nt%dn)!=0),1,"read_wflow_parms [ym2.c]",
                    "ntot must be a multiple of dnms");
      }
      else
      {
         flint=0;
         eps=0.0;
         nt=1;
         dn=1;
      }
   }

   MPI_Bcast(&flint,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&nt,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dn,1,MPI_INT,0,MPI_COMM_WORLD);

   file_head.dn=dn;
   file_head.nn=nt/dn;
   file_head.tmax=N0;
   file_head.eps=eps;

   if ((my_rank==0)&&(noms==0))
   {
      if (append)
      {
         ir=fread(istd,sizeof(stdint_t),3,fdat);
         ir+=fread(dstd,sizeof(double),1,fdat);
         error_root(ir!=4,1,"read_wflow_parms [ym2.c]",
                    "Incorrect read count");

         if (endian==BIG_ENDIAN)
         {
            bswap_int(3,istd);
            bswap_double(1,dstd);
         }

         ie=0;
         ie|=(istd[0]!=(stdint_t)(flint));
         ie|=(istd[1]!=(stdint_t)(nt));
         ie|=(istd[2]!=(stdint_t)(dn));
         ie|=(dstd[0]!=eps);

         error_root(ie!=0,1,"read_wflow_parms [ym2.c]",
                    "Parameters do not match previous run");
      }
      else
      {
         istd[0]=(stdint_t)(flint);
         istd[1]=(stdint_t)(nt);
         istd[2]=(stdint_t)(dn);
         dstd[0]=eps;

         if (endian==BIG_ENDIAN)
         {
            bswap_int(3,istd);
            bswap_double(1,dstd);
         }

         iw=fwrite(istd,sizeof(stdint_t),3,fdat);
         iw+=fwrite(dstd,sizeof(double),1,fdat);
         error_root(iw!=4,1,"read_wflow_parms [ym2.c]",
                    "Incorrect write count");
      }
   }
}


static void read_schedule(void)
{
   int ie,ir,iw;
   stdint_t istd[3];

   if (my_rank==0)
   {
      find_section("Update cycles");
      read_line("nth","%d",&nth);
      read_line("ntot","%d",&ntot);
      read_line("dnlog","%d",&dnlog);
      if (noms==0)
         read_line("dnms","%d",&dnms);
      else
         dnms=0;
      read_line("dncnfg","%d",&dncnfg);

      error_root((append!=0)&&(nth!=0),1,"read_schedule [ym2.c]",
                 "Continuation run: nth must be equal to zero");

      ie=0;
      ie|=(nth<0);
      ie|=(ntot<1);
      ie|=(dnlog<1);
      ie|=(dnlog>dncnfg);
      ie|=((dncnfg%dnlog)!=0);
      ie|=((nth%dncnfg)!=0);
      ie|=((ntot%dncnfg)!=0);

      if (noms==0)
      {
         ie|=(dnms<dnlog);
         ie|=(dnms>dncnfg);
         ie|=((dnms%dnlog)!=0);
         ie|=((dncnfg%dnms)!=0);
      }

      error_root(ie!=0,1,"read_schedule [ym2.c]",
                 "Improper value of nth,ntot,dnlog,dnms or dncnfg");
   }

   MPI_Bcast(&nth,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ntot,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dnlog,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dnms,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&dncnfg,1,MPI_INT,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      if (append)
      {
         ir=fread(istd,sizeof(stdint_t),3,fdat);
         error_root(ir!=3,1,"read_schedule [ym2.c]",
                    "Incorrect read count");

         if (endian==BIG_ENDIAN)
            bswap_int(3,istd);

         ie=0;
         ie|=(istd[0]!=(stdint_t)(dnlog));
         ie|=(istd[1]!=(stdint_t)(dnms));
         ie|=(istd[2]!=(stdint_t)(dncnfg));

         error_root(ie!=0,1,"read_schedule [ym2.c]",
                    "Parameters do not match previous run");
      }
      else
      {
         istd[0]=(stdint_t)(dnlog);
         istd[1]=(stdint_t)(dnms);
         istd[2]=(stdint_t)(dncnfg);

         if (endian==BIG_ENDIAN)
            bswap_int(3,istd);

         iw=fwrite(istd,sizeof(stdint_t),3,fdat);
         error_root(iw!=3,1,"read_schedule [ym2.c]",
                    "Incorrect write count");
      }
   }
}


static void save_iodat(void)
{
   int ir,iw,ie;
   stdint_t istd[6];

   if (my_rank==0)
   {
      if (append)
      {
         ir=fread(istd,sizeof(stdint_t),6,fdat);
         error_root(ir!=6,1,"save_iodat [ym2.c]",
                    "Incorrect read count");

         if (endian==BIG_ENDIAN)
            bswap_int(6,istd);

         ie=0;
         ie|=(istd[0]!=(stdint_t)(iodat[1].type));
         ie|=(istd[1]!=(stdint_t)(iodat[1].nio_nodes));
         ie|=(istd[2]!=(stdint_t)(iodat[1].bs[0]));
         ie|=(istd[3]!=(stdint_t)(iodat[1].bs[1]));
         ie|=(istd[4]!=(stdint_t)(iodat[1].bs[2]));
         ie|=(istd[5]!=(stdint_t)(iodat[1].bs[3]));

         error_root(ie!=0,1,"save_iodat [ym2.c]",
                    "Parameters do not match previous run");
      }
      else
      {
         istd[0]=(stdint_t)(iodat[1].type);
         istd[1]=(stdint_t)(iodat[1].nio_nodes);
         istd[2]=(stdint_t)(iodat[1].bs[0]);
         istd[3]=(stdint_t)(iodat[1].bs[1]);
         istd[4]=(stdint_t)(iodat[1].bs[2]);
         istd[5]=(stdint_t)(iodat[1].bs[3]);

         if (endian==BIG_ENDIAN)
            bswap_int(6,istd);

         iw=fwrite(istd,sizeof(stdint_t),6,fdat);
         error_root(iw!=6,1,"save_iodat [ym2.c]",
                    "Incorrect write count");
      }
   }
}


static void read_infile(int argc,char *argv[])
{
   int ifile,imask,ir;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);

      ifile=find_opt(argc,argv,"-i");
      scnfg=find_opt(argc,argv,"-c");
      append=find_opt(argc,argv,"-a");
      norng=find_opt(argc,argv,"-norng");
      imask=find_opt(argc,argv,"-mask");
      rmmom=find_opt(argc,argv,"-rmmom");
      rmold=find_opt(argc,argv,"-rmold");
      noms=find_opt(argc,argv,"-noms");
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1))||(scnfg==(argc-1))||
                 ((append!=0)&&(scnfg==0)),1,"read_infile [ym2.c]",
                 "Syntax: ym2 -i <filename> "
                 "[-c <filename> [-a [-norng]]|[-mask <int>]] "
                 "[-rmmom] [-rmold] [-noms]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ym2.c]",
                 "Machine has unknown endianness");

      if ((scnfg)&&(!append)&&(imask))
      {
         ir=sscanf(argv[imask+1],"%i",&mask);

         error_root(ir!=1,1,"read_infile [ym2.c]",
                    "Syntax: ym2 -i <filename> "
                    "[-c <filename> [-a [-norng]]|[-mask <int>]] "
                    "[-rmmom] [-rmold] [-noms]");
         error_root((mask<0x0)||(mask>0xf),1,"read_infile [ym2.c]",
                    "Command line argument 'mask' is out of range");
      }
      else
         mask=0x0;

      if (scnfg)
      {
         strncpy(cnfg,argv[scnfg+1],NAME_SIZE-1);
         cnfg[NAME_SIZE-1]='\0';
      }
      else
         cnfg[0]='\0';

      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ym2.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&scnfg,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norng,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&mask,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rmmom,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&rmold,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&noms,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   read_dirs();
   setup_files();

   if (my_rank==0)
   {
      if (append)
         fdat=fopen(par_file,"rb");
      else
         fdat=fopen(par_file,"wb");

      error_root(fdat==NULL,1,"read_infile [ym2.c]",
                 "Unable to open parameter file");
   }

   read_lat_parms();
   read_bc_parms();

   if (my_rank==0)
   {
      if ((append==0)||((iodat[0].type&0x3)&&(norng)))
      {
         find_section("Random number generator");
         read_line("level","%d",&level);
         read_line("seed","%d",&seed);
      }
      else
      {
         level=0;
         seed=0;
      }
   }

   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

   read_smd_parms();
   read_schedule();
   read_integrator();
   read_wflow_parms();
   save_iodat();

   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save);
   }
}


static void check_old_log(int ic,int *nl,int *icnfg)
{
   int ir,isv,irg,lv,sd;
   int np[4],bp[4];

   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [ym2.c]",
              "Unable to open log file");
   (*nl)=0;
   (*icnfg)=0;
   ir=1;
   isv=0;
   irg=0;

   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"process grid")!=NULL)
      {
         ir&=(sscanf(line,"%dx%dx%dx%d process grid, %dx%dx%dx%d",
                     np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8);

         ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                   (np[2]!=NPROC2)||(np[3]!=NPROC3));
         ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                   (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
      }
      else if (strstr(line,"Update cycle no")!=NULL)
      {
         ir&=(sscanf(line,"Update cycle no %d",nl)==1);
         isv=0;
      }
      else if (strstr(line,"Configuration no")!=NULL)
      {
         ir&=(sscanf(line,"Configuration no %d",icnfg)==1);
         isv=1;
      }
      else if (norng)
      {
         if ((strstr(line,"level =")!=NULL)&&(strstr(line,"seed =")!=NULL))
         {
            ir&=(sscanf(line,"level = %d, seed = %d",&lv,&sd)==2);
            irg|=((lv==level)&&(sd==seed));
         }
      }
   }

   fclose(fend);

   error_root(ir!=1,1,"check_old_log [ym2.c]","Incorrect read count");

   error_root(ic!=(*icnfg),1,"check_old_log [ym2.c]",
              "Continuation run:\n"
              "Initial configuration is not the last one of the previous run");

   error_root(isv==0,1,"check_old_log [ym2.c]",
              "Continuation run:\n"
              "The log file extends beyond the last configuration save");

   error_root(irg!=0,1,"check_old_log [ym2.c]",
              "Continuation run:\n"
              "Attempt to reuse previously used ranlux level and seed");
}


static void check_old_dat(int nl)
{
   int nc;
   dat_t ndat;

   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [ym2.c]",
              "Unable to open data file");
   nc=0;

   while (read_dat(1,&ndat)==1)
      nc=ndat.nc;

   fclose(fdat);

   error_root(nc!=nl,1,"check_old_dat [ym2.c]",
              "Continuation run: Incomplete or too many data records");
}


static void check_old_msdat(int nl)
{
   int ic,ir,nc,pnc,dnc;

   fdat=fopen(msdat_file,"rb");
   error_root(fdat==NULL,1,"check_old_msdat [ym2.c]",
              "Unable to open data file");

   check_file_head();

   nc=0;
   dnc=0;
   pnc=0;

   for (ic=0;;ic++)
   {
      ir=read_data();

      if (ir==0)
      {
         error_root(ic==0,1,"check_old_msdat [ym2.c]",
                    "No data records found");
         break;
      }

      nc=data.nc;

      if (ic==1)
      {
         dnc=nc-pnc;
         error_root(dnc<1,1,"check_old_msdat [ym2.c]",
                    "Incorrect update cycle separation");
      }
      else if (ic>1)
         error_root(nc!=(pnc+dnc),1,"check_old_msdat [ym2.c]",
                    "Sequence of update cycles is not equally spaced");

      pnc=nc;
   }

   fclose(fdat);

   error_root((nc!=nl)||((ic>1)&&(dnc!=dnms)),1,
              "check_old_msdat [ym2.c]","Last update cycle numbers "
              "or the update cycle separations do not match");
}


static void check_files(int *nl,int *icnfg)
{
   int ie,ic,icmax,ns[4],bs[4];
   int type,nion,nb,ib,n;

   ipgrd[0]=0;
   ipgrd[1]=0;

   if (my_rank==0)
   {
      if (append)
      {
         error_root(strstr(cnfg,nbase)!=cnfg,1,
                    "check_files [ym2.c]","Continuation run:\n"
                    "Run name does not match the previous one");
         error_root(sscanf(cnfg+strlen(nbase),"n%d",&ic)!=1,1,
                    "check_files [ym2.c]","Continuation run:\n"
                    "Unable to read configuration number from file name");
         error_root(((iodat[0].type&iodat[1].type)==0x0)||
                    ((iodat[0].type&0x6)&&
                     (iodat[0].nio_nodes!=iodat[1].nio_nodes)),1,
                    "check_files [ym2.c]","Continuation run:\n"
                    "Unexpected initial configuration storage type");

         check_old_log(ic,nl,icnfg);
         check_old_dat(*nl);
         if (noms==0)
            check_old_msdat(*nl);

         (*icnfg)+=1;
      }
      else
      {
         ie=check_file(log_file,"r");
         ie|=check_file(dat_file,"rb");

         if (noms==0)
            ie|=check_file(msdat_file,"rb");

         error_root(ie!=0,1,"check_files [ym2.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         if (noms==0)
         {
            fdat=fopen(msdat_file,"wb");
            error_root(fdat==NULL,1,"check_files [ym2.c]",
                       "Unable to open measurement data file");
            write_file_head();
            fclose(fdat);
         }

         (*nl)=0;
         (*icnfg)=1;
      }
   }

   MPI_Bcast(nl,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(icnfg,1,MPI_INT,0,MPI_COMM_WORLD);
   icmax=(*icnfg)+(ntot-nth)/dncnfg;

   if (scnfg)
   {
      type=iodat[0].type;

      if (type&0x1)
      {
         error(name_size("%s/%s",init_dir,cnfg)>=NAME_SIZE,1,
               "check_files [ym2.c]","init_dir name is too long");
         check_dir_root(init_dir);
      }
      else if (type&0x2)
      {
         error(name_size("%s/0/0/%s_b0",init_dir,cnfg)>=NAME_SIZE,1,
               "check_files [ym2.c]","init_dir name is too long");
         sprintf(line,"%s/0/0",init_dir);
         if ((cpr[0]==0)&&(cpr[1]==0)&&(cpr[2]==0)&&(cpr[3]==0))
            check_dir(line);

         sprintf(line,"%s/0/0/%s_b0",init_dir,cnfg);
         blk_sizes(line,ns,bs);

         if (append)
         {
            error_root((bs[0]!=iodat[1].bs[0])||(bs[1]!=iodat[1].bs[1])||
                       (bs[2]!=iodat[1].bs[2])||(bs[3]!=iodat[1].bs[3]),1,
                       "check_files [ym2.c]","Continuation run:\n"
                       "Unexpected initial configuration storage block size");
         }

         ib=blk_index(ns,bs,&nb);
         nion=iodat[0].nio_nodes;
         n=nb/nion;
         error_root(nb%nion!=0,1,"check_files [ym2.c]",
                    "Number of blocks is not a multiple of nio_nodes");
         error(name_size("%s/%d/%d/%s_b%d",init_dir,nion-1,n-1,
                         cnfg,nb-1)>=NAME_SIZE,1,
               "check_files [ym2.c]","init_dir name is too long");
         sprintf(line,"%s/%d/%d",init_dir,ib/n,ib%n);
         strcpy(init_dir,line);
         if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
             ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0)&&(ib<nb))
            check_dir(init_dir);

         iodat[0].nb=nb;
         iodat[0].ib=ib;
         iodat[0].bs[0]=bs[0];
         iodat[0].bs[1]=bs[1];
         iodat[0].bs[2]=bs[2];
         iodat[0].bs[3]=bs[3];
      }
      else if (type&0x4)
      {
         nion=iodat[0].nio_nodes;
         n=NPROC/nion;
         error_root(nb%nion!=0,1,"check_files [ym2.c]",
                    "Number of processes is not a multiple of nio_nodes");
         error(name_size("%s/%d/%d/%s_%d",loc_dir,nion-1,n-1,
                         cnfg,NPROC-1)>=NAME_SIZE,1,
               "check_files [ym2.c]","init_dir name is too long");
         sprintf(line,"%s/%d/%d",init_dir,my_rank/n,my_rank%n);
         strcpy(init_dir,line);
         check_dir(init_dir);
      }
   }

   type=iodat[1].type;

   if (type&0x1)
   {
      error(name_size("%s/%sn%d",cnfg_dir,nbase,icmax)>=NAME_SIZE,1,
            "check_files [ym2.c]","cnfg_dir name is too long");
      check_dir_root(cnfg_dir);
   }

   if (type&0x2)
   {
      ns[0]=N0;
      ns[1]=N1;
      ns[2]=N2;
      ns[3]=N3;
      bs[0]=iodat[1].bs[0];
      bs[1]=iodat[1].bs[1];
      bs[2]=iodat[1].bs[2];
      bs[3]=iodat[1].bs[3];
      ib=blk_index(ns,bs,&nb);

      nion=iodat[1].nio_nodes;
      n=nb/nion;
      error_root(nb%nion!=0,1,"check_files [ym2.c]",
                 "Number of blocks is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_b%d",blk_dir,nion-1,n-1,nbase,
                      icmax,nb-1)>=NAME_SIZE,1,
            "check_files [ym2.c]","block_dir name is too long");
      sprintf(line,"%s/%d/%d",blk_dir,ib/n,ib%n);
      strcpy(blk_dir,line);
      if (((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
          ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0))
         check_dir(blk_dir);

      iodat[1].nb=nb;
      iodat[1].ib=ib;
   }

   if (type&0x4)
   {
      nion=iodat[1].nio_nodes;
      n=NPROC/nion;
      error_root(NPROC%nion!=0,1,"check_files [ym2.c]",
                 "Number of processes is not a multiple of nio_nodes");
      error(name_size("%s/%d/%d/%sn%d_%d",loc_dir,nion-1,n-1,nbase,
                      icmax,NPROC-1)>=NAME_SIZE,1,
            "check_files [ym2.c]","loc_dir name is too long");
      sprintf(line,"%s/%d/%d",loc_dir,my_rank/n,my_rank%n);
      strcpy(loc_dir,line);
      check_dir(loc_dir);
   }
}


static void init_rng(int icnfg)
{
   int ic;

   if (append)
   {
      if (iodat[0].type&0x3)
      {
         if (norng)
            start_ranlux(level,seed);
         else
         {
            ic=import_ranlux(rng_file);
            error_root(ic!=(icnfg-1),1,"init_rng [ym2.c]",
                       "Configuration number mismatch (*.rng file)");
         }
      }
   }
   else
      start_ranlux(level,seed);
}


static void init_flds(void)
{
   int type;
   double wt1,wt2;

   if (scnfg)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      type=iodat[0].type;

      if (type&0x1)
      {
         sprintf(cnfg_file,"%s/%s",init_dir,cnfg);
         import_cnfg(cnfg_file,mask);

         if (append)
         {
            sprintf(cnfg_file,"%s/%s.mom",init_dir,cnfg);
            import_mfld(cnfg_file);
         }
         else
            random_mom();
      }
      else if (type&0x2)
      {
         set_nio_streams(iodat[0].nio_streams);
         sprintf(cnfg_file,"%s/%s_b%d",init_dir,cnfg,iodat[0].ib);
         blk_import_cnfg(cnfg_file,mask);

         if (append)
         {
            sprintf(cnfg_file,"%s/%s.mom_b%d",init_dir,cnfg,iodat[0].ib);
            blk_import_mfld(cnfg_file);
         }
         else
            random_mom();
      }
      else
      {
         set_nio_streams(iodat[0].nio_streams);
         sprintf(cnfg_file,"%s/%s_%d",init_dir,cnfg,my_rank);
         read_cnfg(cnfg_file);

         if (append)
         {
            sprintf(cnfg_file,"%s/%s.mom_%d",init_dir,cnfg,my_rank);
            read_mfld(cnfg_file);
         }
         else
            random_mom();
      }

      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      if (my_rank==0)
      {
         printf("Initial configuration read from disk in %.2e sec\n\n",
                wt2-wt1);
         fflush(flog);
      }
   }
   else
   {
      random_ud();
      random_mom();
   }

   if (iodat[1].type&0x6)
      set_nio_streams(iodat[1].nio_streams);
}


static void store_ud(su3_dble *usv)
{
   su3_dble *udb;

   udb=udfld();
   cm3x3_assign(4*VOLUME,udb,usv);
}


static void recall_ud(su3_dble *usv)
{
   su3_dble *udb;

   udb=udfld();
   cm3x3_assign(4*VOLUME,usv,udb);
   set_flags(UPDATED_UD);
}


static void set_data(int nc)
{
   int in,dn,nn;
   double eps;

   data.nc=nc;
   dn=file_head.dn;
   nn=file_head.nn;
   eps=file_head.eps;

   for (in=0;in<nn;in++)
   {
      Wact[in]=plaq_action_slices(data.Wsl[in]);
      Yact[in]=ym_action_slices(data.Ysl[in]);
      Qtop[in]=tcharge_slices(data.Qsl[in]);

      if (flint==0)
         fwd_euler(dn,eps);
      else if (flint==1)
         fwd_rk2(dn,eps);
      else
         fwd_rk3(dn,eps);
   }

   Wact[in]=plaq_action_slices(data.Wsl[in]);
   Yact[in]=ym_action_slices(data.Ysl[in]);
   Qtop[in]=tcharge_slices(data.Qsl[in]);
}


static void print_info(int icnfg)
{
   int n,type;
   long ip;
   mdint_parms_t mdp;

   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");

      if (append)
         flog=freopen(log_file,"a",stdout);
      else
         flog=freopen(log_file,"w",stdout);

      error_root(flog==NULL,1,"print_info [ym2.c]","Unable to open log file");

      if (append)
         printf("Continuation run, start from configuration %s\n\n",cnfg);
      else
      {
         printf("\n");
         printf("Simulation of the SU(3) gauge theory\n");
         printf("------------------------------------\n\n");

         if (scnfg)
         {
            if (iodat[0].type&0x3)
               printf("New run, start from configuration %s "
                      "(extension mask = %#x)\n\n",cnfg,(unsigned int)(mask));
            else
               printf("New run, start from configuration %s\n\n",cnfg);
         }
         else
            printf("New run, start from random configuration\n\n");

         printf("Using the SMD algorithm\n");
         printf("Program version %s\n",openQCD_RELEASE);
      }

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");

      if ((ipgrd[0]!=0)&&(ipgrd[1]!=0))
         printf("Process grid and process block size changed:\n");
      else if (ipgrd[0]!=0)
         printf("Process grid changed:\n");
      else if (ipgrd[1]!=0)
         printf("Process block size changed:\n");

      if ((append==0)||(ipgrd[0]!=0)||(ipgrd[1]!=0))
      {
         printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d process block size\n\n",
                NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);
      }

      if (append==0)
      {
         n=fdigits(lat.beta);
         printf("beta = %.*f\n",IMAX(n,1),lat.beta);
         n=fdigits(lat.c0);
         printf("c0 = %.*f, ",IMAX(n,1),lat.c0);
         n=fdigits(lat.c1);
         printf("c1 = %.*f\n",IMAX(n,1),lat.c1);
         print_bc_parms(1);
      }

      printf("Random number generator:\n");

      if (append)
      {
         if (iodat[0].type&0x3)
         {
            if (norng)
               printf("level = %d, seed = %d\n\n",level,seed);
            else
            {
               printf("State of ranlxs and ranlxd reset to the\n");
               printf("last exported state\n\n");
            }
         }
         else
         {
            printf("State of ranlxs and ranlxd read from\n");
            printf("initial field-configuration file\n\n");
         }
      }
      else
         printf("level = %d, seed = %d\n\n",level,seed);

      if (append)
      {
         printf("Update cycles:\n");
         printf("ntot = %d\n\n",ntot);
      }
      else
      {
         printf("Update cycles:\n");
         n=fdigits(smd.gamma);
         printf("gamma = %.*f\n",IMAX(n,1),smd.gamma);
         n=fdigits(smd.eps);
         printf("eps = %.*f\n",IMAX(n,1),smd.eps);

         mdp=mdint_parms(0);

         if (mdp.integrator==LPFR)
            printf("Leapfrog integrator\n");
         else if (mdp.integrator==OMF2)
            printf("2nd order OMF integrator with lambda = %.4e\n",mdp.lambda);
         else if (mdp.integrator==OMF4)
            printf("4th order OMF integrator\n");

         printf("Number of steps = %d\n",mdp.nstep);

         if (smd.iacc)
            printf("Acceptance-rejection step switched on\n\n");
         else
            printf("Acceptance-rejection step switched off\n\n");

         printf("nth = %d, ntot = %d\n",nth,ntot);

         if (noms)
         {
            printf("dnlog = %d, dncnfg = %d\n\n",
                   dnlog,dncnfg);
            printf("Wilson flow observables are not measured\n\n");
         }
         else
         {
            printf("dnlog = %d, dnms = %d, dncnfg = %d\n\n",
                   dnlog,dnms,dncnfg);
            printf("Online measurement of Wilson flow observables\n\n");

            printf("Wilson flow:\n");
            if (flint==0)
               printf("Euler integrator\n");
            else if (flint==1)
               printf("2nd order RK integrator\n");
            else
               printf("3rd order RK integrator\n");
            n=fdigits(file_head.eps);
            printf("eps = %.*f\n",IMAX(n,1),file_head.eps);
            printf("ntot = %d\n",file_head.dn*file_head.nn);
            printf("dnms = %d\n\n",file_head.dn);
         }
      }

      if (scnfg)
      {
         printf("Initial configuration storage type = ");
         type=iodat[0].type;

         if (type&0x1)
            printf("exported\n");
         else if (type&0x2)
            printf("block-exported\n");
         else
            printf("local\n");

         if (type&0x6)
            printf("Parallel configuration input: "
                   "nio_nodes = %d, nio_streams = %d\n",
                   iodat[0].nio_nodes,iodat[0].nio_streams);
         printf("\n");
      }

      printf("The generated configurations are ");
      type=iodat[1].type;

      if (type==0x1)
         printf("exported\n");
      else if (type==0x2)
         printf("block-exported\n");
      else if (type==0x3)
         printf("exported and block-exported\n");
      else if (type==0x4)
         printf("locally stored\n");
      else if (type==0x5)
         printf("exported and locally stored\n");
      else if (type==0x6)
         printf("block-exported and locally stored\n");
      else
         printf("exported, block-exported and locally stored\n");

      if (type&0x6)
      {
         printf("Parallel configuration output: "
                "nio_nodes = %d, nio_streams = %d",
                iodat[1].nio_nodes,iodat[1].nio_streams);
         if (type&0x2)
            printf(", bs = %dx%dx%dx%d",iodat[1].bs[0],iodat[1].bs[1],
                   iodat[1].bs[2],iodat[1].bs[3]);
         printf("\n");
      }

      printf("\n");

      if (rmold)
         printf("Old gauge and momentum configurations are deleted\n\n");
      else if (rmmom)
         printf("Old momentum configurations are deleted\n\n");

      fflush(flog);
   }
}


static void print_log(dat_t *ndat)
{
   if (my_rank==0)
   {
      printf("Update cycle no %d\n",(*ndat).nc);
      if (smd.iacc)
         printf("dH = %+.1e, iac = %d\n",(*ndat).dH,(*ndat).iac);
      else
         printf("dH = %+.1e\n",(*ndat).dH);
      printf("Average plaquette = %.6f\n",(*ndat).avpl);
   }
}


static void save_dat(int n,double siac,double wtcyc,double wtall,dat_t *ndat)
{
   int iw;

   if (my_rank==0)
   {
      fdat=fopen(dat_file,"ab");
      error_root(fdat==NULL,1,"save_dat [ym2.c]",
                 "Unable to open data file");

      iw=write_dat(1,ndat);
      error_root(iw!=1,1,"save_dat [ym2.c]",
                 "Incorrect write count");
      fclose(fdat);

      if (smd.iacc)
         printf("Acceptance rate = %.6f\n",siac/(double)(n));
      printf("Time per update cycle = %.2e sec (average = %.2e sec)\n\n",
             wtcyc/(double)(dnlog),wtall/(double)(n));
      fflush(flog);
   }
}


static void save_msdat(int n,double wtms,double wtmsall)
{
   int nms,in,dn,nn,din;
   double eps;

   if (my_rank==0)
   {
      fdat=fopen(msdat_file,"ab");
      error_root(fdat==NULL,1,"save_msdat [ym2.c]",
                 "Unable to open data file");
      write_data();
      fclose(fdat);

      nms=(n-nth)/dnms+(nth>0);
      dn=file_head.dn;
      nn=file_head.nn;
      eps=file_head.eps;

      din=nn/10;
      if (din<1)
         din=1;

      printf("Measurement run:\n\n");

      for (in=0;in<=nn;in+=din)
         printf("n = %3d, t = %.2e, Wact = %.6e, Yact = %.6e, Q = % .2e\n",
                in*dn,eps*(double)(in*dn),Wact[in],Yact[in],Qtop[in]);

      printf("\n");
      printf("Configuration fully processed in %.2e sec ",wtms);
      printf("(average = %.2e sec)\n",wtmsall/(double)(nms));
      printf("Measured data saved\n\n");
      fflush(flog);
   }
}


static void save_flds(int icnfg)
{
   int type,ie;
   double wt1,wt2;

   MPI_Barrier(MPI_COMM_WORLD);
   wt1=MPI_Wtime();

   ie=(query_flags(UD_PHASE_SET)||(check_bc(0.0)==0));
   error_root(ie!=0,1,"save_flds [ym2.c]",
              "Phase-modified field or unexpected boundary values");
   type=iodat[1].type;

   if (type&0x1)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      export_cnfg(cnfg_file);
      sprintf(cnfg_file,"%s/%sn%d.mom",cnfg_dir,nbase,icnfg);
      export_mfld(cnfg_file);
   }

   if (type&0x2)
   {
      sprintf(cnfg_file,"%s/%sn%d_b%d",blk_dir,nbase,icnfg,iodat[1].ib);
      blk_export_cnfg(iodat[1].bs,cnfg_file);
      sprintf(cnfg_file,"%s/%sn%d.mom_b%d",blk_dir,nbase,icnfg,iodat[1].ib);
      blk_export_mfld(iodat[1].bs,cnfg_file);
   }

   if (type&0x4)
   {
      sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,icnfg,my_rank);
      write_cnfg(cnfg_file);
      sprintf(cnfg_file,"%s/%sn%d.mom_%d",loc_dir,nbase,icnfg,my_rank);
      write_mfld(cnfg_file);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   wt2=MPI_Wtime();

   if (my_rank==0)
   {
      printf("Configuration no %d ",icnfg);

      if (type==0x1)
         printf("exported");
      else if (type==0x2)
         printf("block-exported");
      else if (type==0x3)
         printf("exported and block-exported");
      else if (type==0x4)
         printf("locally stored");
      else if (type==0x5)
         printf("exported and locally stored");
      else if (type==0x6)
         printf("block-exported and locally stored");
      else
         printf("exported, block-exported and locally stored");

      printf(" in %.2e sec\n\n",wt2-wt1);
   }
}


static void check_endflag(int *iend)
{
   if (my_rank==0)
   {
      fend=fopen(end_file,"r");

      if (fend!=NULL)
      {
         fclose(fend);
         remove(end_file);
         (*iend)=1;
         printf("End flag set, run stopped\n\n");
      }
      else
         (*iend)=0;
   }

   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


static void remove_flds(int icnfg)
{
   int type,nios,n,i;
   int nb,ib,ip,*bs;

   if (((rmold)||(rmmom))&&(icnfg>=1))
   {
      MPI_Barrier(MPI_COMM_WORLD);
      type=iodat[1].type;

      if ((type&0x1)&&(my_rank==0))
      {
         if (rmold)
         {
            sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
            remove(cnfg_file);
         }

         sprintf(cnfg_file,"%s/%sn%d.mom",cnfg_dir,nbase,icnfg);
         remove(cnfg_file);
      }

      if (type&0x2)
      {
         nios=iodat[1].nio_streams;
         nb=iodat[1].nb;
         ib=iodat[1].ib;
         bs=iodat[1].bs;
         ip=(((cpr[0]*L0)%bs[0]==0)&&((cpr[1]*L1)%bs[1]==0)&&
             ((cpr[2]*L2)%bs[2]==0)&&((cpr[3]*L3)%bs[3]==0));
         n=nb/nios;

         for (i=0;i<n;i++)
         {
            if ((i==(ib%n))&&(ip!=0))
            {
               if (rmold)
               {
                  sprintf(cnfg_file,"%s/%sn%d_b%d",blk_dir,nbase,icnfg,ib);
                  remove(cnfg_file);
               }

               sprintf(cnfg_file,"%s/%sn%d.mom_b%d",blk_dir,nbase,icnfg,ib);
               remove(cnfg_file);
            }

            MPI_Barrier(MPI_COMM_WORLD);
         }
      }

      if (type&0x4)
      {
         nios=iodat[1].nio_streams;
         n=NPROC/nios;

         for (i=0;i<n;i++)
         {
            if (i==(my_rank%n))
            {
               if (rmold)
               {
                  sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,icnfg,my_rank);
                  remove(cnfg_file);
               }

               sprintf(cnfg_file,"%s/%sn%d.mom_%d",loc_dir,nbase,icnfg,my_rank);
               remove(cnfg_file);
            }

            MPI_Barrier(MPI_COMM_WORLD);
         }
      }
   }
}


int main(int argc,char *argv[])
{
   int nl,icnfg;
   int n,nc,iend,iac,i;
   double npl,siac,*qsm[1];
   double wt1,wt2,wtcyc,wtall,wtms,wtmsall;
   qflt act0[2],act1[2];
   su3_dble **usv;
   dat_t ndat;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   check_machine();
   if (noms==0)
      alloc_data();
   geometry();
   check_files(&nl,&icnfg);
   print_info(icnfg);

   if ((smd.iacc)||(noms==0))
      alloc_wud(1);

   if ((smd.iacc)||((noms==0)&&(flint)))
      alloc_wfd(1);

   set_mdsteps();
   if (!append)
      print_msize(1);
   init_rng(icnfg);
   init_flds();

   if (bc_type()==0)
      npl=(double)(6*(N0-1)*N1)*(double)(N2*N3);
   else
      npl=(double)(6*N0*N1)*(double)(N2*N3);

   iend=0;
   iac=1;
   siac=0.0;
   wtcyc=0.0;
   wtall=0.0;
   wtms=0.0;
   wtmsall=0.0;

   for (n=0;(iend==0)&&(n<ntot);)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();

      for (nc=0;nc<dnlog;nc++)
      {
         if (smd.iacc)
         {
            iac=run_smd(act0,act1);
            siac+=(double)(iac);
         }
         else
         {
            if (nc==(dnlog-1))
               run_smd_noacc0(act0,act1);
            else
               run_smd_noacc1();
         }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      n+=dnlog;
      wt2=MPI_Wtime();
      wtcyc+=(wt2-wt1);
      wtall+=wtcyc;

      for (i=0;i<2;i++)
      {
         act0[i].q[0]=-act0[i].q[0];
         act0[i].q[1]=-act0[i].q[1];
         add_qflt(act0[i].q,act1[i].q,act1[i].q);
         if (i>0)
            add_qflt(act1[i].q,act1[0].q,act1[0].q);
      }

      qsm[0]=act1[0].q;
      global_qsum(1,qsm,qsm);

      ndat.nc=nl+n;
      ndat.iac=iac;
      ndat.dH=act1[0].q[0];
      ndat.avpl=plaq_wsum_dble(1)/npl;

      print_log(&ndat);
      save_dat(n,siac,wtcyc,wtall,&ndat);
      wtcyc=0.0;

      if ((noms==0)&&(n>=nth)&&((n%dnms)==0))
      {
         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();

         usv=reserve_wud(1);
         store_ud(usv[0]);
         set_data(nl+n);
         recall_ud(usv[0]);
         release_wud();

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();

         wtms=wt2-wt1;
         wtmsall+=wtms;
         save_msdat(n,wtms,wtmsall);
      }

      if ((n>=nth)&&((n%dncnfg)==0))
      {
         save_flds(icnfg);
         export_ranlux(icnfg,rng_file);
         check_endflag(&iend);

         if (my_rank==0)
         {
            fflush(flog);
            copy_file(log_file,log_save);
            copy_file(dat_file,dat_save);
            if (noms==0)
               copy_file(msdat_file,msdat_save);
            copy_file(rng_file,rng_save);
         }

         remove_flds(icnfg-1);
         icnfg+=1;
      }
   }

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
