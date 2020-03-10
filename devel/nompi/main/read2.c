/*******************************************************************************
*
* File read2.c
*
* Copyright (C) 2012-2014, 2018, 2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reads and evaluates the data on the data files <name>.ms1.dat created by
* the program ms1.
*
* The program writes the history of the measured normalized reweighting
* factors to the file <name>.rw1.dat in the plots directory and prints
* the associated integrated autocorrelation times to stdout.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "su3.h"
#include "utils.h"
#include "analysis.h"

static struct
{
   int nrw;
   int *nfct,*nsrc;
} file_head;

static struct
{
   array_t *rw,*rwall;
   array_t **sqn,**lnr,**lnrw;
} arrays;

static struct
{
   int nc;
   qflt ***sqn,***lnr;
} data;

static int endian;
static int first,last,step,nms;
static double *rwall,**rw;
static qflt ****lnrw;


static void read_file_head(FILE *fdat)
{
   int nrw,ndmy,iw,*nfct;
   double *rdmy;

   iw=read_parms(fdat,&nrw,&nfct,&ndmy,&rdmy);

   error((nrw<2)||((nrw%2)!=0)||(iw!=2+nrw)||(ndmy!=0)||(rdmy!=NULL),1,
         "read_file_head [read2.c]","Unexpected parameter data");

   nrw/=2;
   file_head.nrw=nrw;
   file_head.nfct=nfct;
   file_head.nsrc=nfct+nrw;
}


static void alloc_data(void)
{
   int nrw,irw,*nfct,*nsrc;
   size_t n[2];
   qflt ***qp;
   array_t **ap;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;

   qp=malloc(2*nrw*sizeof(*qp));
   ap=malloc(2*nrw*sizeof(*ap));
   error((qp==NULL)||(ap==NULL),1,"alloc_data [read2.c]",
         "Unable to allocate data arrays");
   data.sqn=qp;
   data.lnr=qp+nrw;
   arrays.sqn=ap;
   arrays.lnr=ap+nrw;

   for (irw=0;irw<nrw;irw++)
   {
      n[0]=nfct[irw];
      n[1]=2*nsrc[irw];
      arrays.sqn[irw]=alloc_array(2,n,sizeof(double),0);
      arrays.lnr[irw]=alloc_array(2,n,sizeof(double),0);
      data.sqn[irw]=(qflt**)(arrays.sqn[irw][0].a);
      data.lnr[irw]=(qflt**)(arrays.lnr[irw][0].a);
   }
}


static int read_data(FILE *fdat)
{
   int nrw,irw,ir;
   stdint_t istd[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   data.nc=(int)(istd[0]);
   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      read_array(fdat,arrays.sqn[irw]);
      read_array(fdat,arrays.lnr[irw]);
   }

   return 1;
}


static void cnfg_range(FILE *fdat,int *fst,int *lst,int *stp)
{
   int nc,ie;

   (*fst)=0;
   (*lst)=0;
   (*stp)=1;
   nc=0;
   ie=0;

   while (read_data(fdat))
   {
      nc+=1;

      if (nc==1)
         (*fst)=data.nc;
      else if (nc==2)
         (*stp)=data.nc-(*fst);
      else
         ie|=((data.nc-(*lst))!=(*stp));

      (*lst)=data.nc;
   }

   error(nc==0,1,"cnfg_range [read2.c]","No data records on data file");
   error(ie!=0,1,"cnfg_range [read2.c]","Non-contiguous configuration numbers");
}


static void select_cnfg_range(FILE *fdat)
{
   int fst,lst,stp;

   cnfg_range(fdat,&fst,&lst,&stp);

   printf("Available configuration range: %d - %d by %d\n",
          fst,lst,stp);
   printf("Select first,last,step: ");
   scanf("%d",&first);
   scanf(",");
   scanf("%d",&last);
   scanf(",");
   scanf("%d",&step);
   printf("\n");

   error((step<=0)||((step%stp)!=0),1,"select_cnfg_range [read2.c]",
         "Step must be positive and divisible by the configuration separation");

   if (first<fst)
   {
      first=first+((fst-first)/step)*step;
      if (first<fst)
         first+=step;
   }

   error((first>lst)||(((first-fst)%stp)!=0)||(last<first),1,
         "select_cnfg_range [read2.c]","Improper configuration range");

   if (last>lst)
      last=lst;
   last=last-(last-first)%step;

   printf("Selected configuration range: %d - %d by %d.\n",
          first,last,step);
   nms=(last-first)/step+1;
}


static void read_lnrw(FILE *fdat)
{
   int nrw,*nfct,*nsrc;
   int irw,ifct,isrc,ims,nc;
   size_t n[3];
   qflt ****qp;
   array_t **ap;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;

   qp=malloc(nrw*sizeof(*qp));
   ap=malloc(nrw*sizeof(*ap));
   error((qp==NULL)||(ap==NULL),1,"read_lnrw [read2.c]",
         "Unable to allocate data arrays");
   lnrw=qp;
   arrays.lnrw=ap;

   for (irw=0;irw<nrw;irw++)
   {
      n[0]=nfct[irw];
      n[1]=nsrc[irw];
      n[2]=2*nms;
      arrays.lnrw[irw]=alloc_array(3,n,sizeof(double),0);
      lnrw[irw]=(qflt***)(arrays.lnrw[irw][0].a);
   }

   ims=0;

   while (read_data(fdat))
   {
      nc=data.nc;

      if ((nc>=first)&&(nc<=last)&&(((nc-first)%step)==0))
      {
         for (irw=0;irw<nrw;irw++)
         {
            for (ifct=0;ifct<nfct[irw];ifct++)
            {
               for (isrc=0;isrc<nsrc[irw];isrc++)
               {
                  lnrw[irw][ifct][isrc][ims].q[0]=
                     -data.lnr[irw][ifct][isrc].q[0];
                  lnrw[irw][ifct][isrc][ims].q[1]=
                     -data.lnr[irw][ifct][isrc].q[1];
               }
            }
         }

         ims+=1;
      }
   }

   error(ims!=nms,1,"read_lnrw [read2.c]","Incomplete data or read error");
}


static void shift_lnrw(void)
{
   int nrw,*nfct,*nsrc;
   int irw,ifct,isrc,ims;
   double lnav,*q;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;

   for (irw=0;irw<nrw;irw++)
   {
      for (ifct=0;ifct<nfct[irw];ifct++)
      {
         lnav=0.0;

         for (isrc=0;isrc<nsrc[irw];isrc++)
         {
            for (ims=0;ims<nms;ims++)
            {
               q=lnrw[irw][ifct][isrc][ims].q;
               lnav+=q[0];
            }
         }

         lnav/=(double)(nms*nsrc[irw]);
         lnav=-lnav;

         for (isrc=0;isrc<nsrc[irw];isrc++)
         {
            for (ims=0;ims<nms;ims++)
            {
               q=lnrw[irw][ifct][isrc][ims].q;
               acc_qflt(lnav,q);
            }
         }
      }
   }
}


static void normalize_array(int n,double *r)
{
   int i;
   double s;
   qflt sm;

   sm.q[0]=0.0;
   sm.q[1]=0.0;

   for (i=0;i<n;i++)
      acc_qflt(r[i],sm.q);

   s=(double)(n)/sm.q[0];

   for (i=0;i<n;i++)
      r[i]*=s;
}


static void set_rw(void)
{
   int nrw,*nfct,*nsrc;
   int irw,ifct,isrc,ims;
   size_t n[2];
   double ra,r;

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;

   n[0]=nrw;
   n[1]=nms;
   arrays.rw=alloc_array(2,n,sizeof(double),0);
   rw=(double**)(arrays.rw[0].a);

   for (irw=0;irw<nrw;irw++)
   {
      for (ims=0;ims<nms;ims++)
      {
         ra=1.0;

         for (ifct=0;ifct<nfct[irw];ifct++)
         {
            r=0.0;

            for (isrc=0;isrc<nsrc[irw];isrc++)
               r+=exp(lnrw[irw][ifct][isrc][ims].q[0]);

            r/=(double)(nsrc[irw]);
            ra*=r;
         }

         rw[irw][ims]=ra;
      }

      normalize_array(nms,rw[irw]);
   }
}


static void set_rwall(void)
{
   int nrw,irw,ims;
   size_t n[1];

   n[0]=nms;
   arrays.rwall=alloc_array(1,n,sizeof(double),0);
   rwall=(double*)(arrays.rwall[0].a);
   nrw=file_head.nrw;

   for (ims=0;ims<nms;ims++)
   {
      rwall[ims]=1.0;

      for (irw=0;irw<nrw;irw++)
         rwall[ims]*=rw[irw][ims];
   }

   normalize_array(nms,rwall);
}


static void set_rwstat(int irw,double *rws)
{
   int nrw,ims;
   double rmn,rmx,sig,*r;

   nrw=file_head.nrw;

   if (irw<nrw)
      r=rw[irw];
   else
      r=rwall;

   rmn=r[0];
   rmx=r[0];
   sig=0.0;

   for (ims=0;ims<nms;ims++)
   {
      if (r[ims]<rmn)
         rmn=r[ims];
      if (r[ims]>rmx)
         rmx=r[ims];

      sig+=(r[ims]-1.0)*(r[ims]-1.0);
   }

   sig=sqrt(sig/(double)(nms));

   rws[0]=rmn;
   rws[1]=rmx;
   rws[2]=sig;
}


static void read_data_file(char *fin)
{
   long ipos;
   FILE *fdat;

   fdat=fopen(fin,"rb");
   error(fdat==NULL,1,"read_data_file [read2.c]","Unable to open file");
   printf("Read data from file %s\n\n",fin);

   endian=endianness();
   read_file_head(fdat);
   alloc_data();

   ipos=ftell(fdat);
   select_cnfg_range(fdat);

   fseek(fdat,ipos,SEEK_SET);
   read_lnrw(fdat);
   fclose(fdat);
}


static void print_plot(char *fin)
{
   int n,nrw,irw,ims;
   char base[NAME_SIZE],plt_file[NAME_SIZE],*p;
   double rwstat[3];
   FILE *fout;

   p=strstr(fin,".ms1.dat");
   error(p==NULL,1,"print_plot [read2.c]","Unexpected data file name");
   n=p-fin;

   p=strrchr(fin,'/');
   if (p==NULL)
      p=fin;
   else
      p+=1;
   n-=(p-fin);

   error(n>=NAME_SIZE,1,"print_plot [read2.c]","File name is too long");
   strncpy(base,p,n);
   base[n]='\0';

   error(name_size("plots/%s.rw1.dat",base)>=NAME_SIZE,1,
         "print_plot [read2.c]","File name is too long");
   sprintf(plt_file,"plots/%s.rw1.dat",base);
   fout=fopen(plt_file,"w");
   error(fout==NULL,1,"print_plot [read2.c]",
         "Unable to open output file");

   nrw=file_head.nrw;

   fprintf(fout,"#\n");
   fprintf(fout,"# Data written by the program ms1\n");
   fprintf(fout,"# -------------------------------\n");
   fprintf(fout,"#\n");
   fprintf(fout,"# Number of measurements = %d\n",nms);
   fprintf(fout,"#\n");
   fprintf(fout,"# nc:   Configuration number\n");
   fprintf(fout,"# W:    Normalized reweighting factors\n");
   fprintf(fout,"#\n");
   fprintf(fout,"# Minimal value, maximal value, standard deviation:\n");

   for (irw=0;irw<nrw;irw++)
   {
      set_rwstat(irw,rwstat);

      fprintf(fout,"# W[%d]:   %.2e, %.2e, %.2e\n",
              irw,rwstat[0],rwstat[1],rwstat[2]);
   }

   if (nrw>1)
   {
      set_rwstat(nrw,rwstat);
      fprintf(fout,"# W[all]: %.2e, %.2e, %.2e\n",
              rwstat[0],rwstat[1],rwstat[2]);
   }

   fprintf(fout,"#\n");
   fprintf(fout,"#  nc");

   for (irw=0;irw<nrw;irw++)
      fprintf(fout,"       W[%d] ",irw);

   if (nrw==1)
      fprintf(fout,"\n");
   else
      fprintf(fout,"       W[all]\n");

   fprintf(fout,"#\n");

   for (ims=0;ims<nms;ims++)
   {
      fprintf(fout," %5d  ",first+ims*step);

      for (irw=0;irw<nrw;irw++)
         fprintf(fout,"  %.4e",rw[irw][ims]);

      if (nrw==1)
         fprintf(fout,"\n");
      else
         fprintf(fout,"  %.4e\n",rwall[ims]);
   }

   fclose(fout);

   printf("Data printed to file %s\n\n",plt_file);
}


static double f(int nx,double *x)
{
   return x[0];
}


int main(int argc,char *argv[])
{
   int nrw,irw,*nfct,*nsrc;

   error(argc!=2,1,"main [read2.c]","Syntax: read2 <filename>");

   printf("\n");
   printf("History of reweighting factors\n");
   printf("------------------------------\n\n");

   read_data_file(argv[1]);

   printf("The total number of measurements is %d.\n",nms);
   printf("Integrated autocorrelation times and associated errors are ");
   printf("estimated\n");

   if (nms>100)
      printf("using the numerically determined autocorrelation function.\n");
   else
      printf("by binning and calculating jackknife errors.\n");

   printf("Autocorrelation times are given in numbers of measurements.\n\n");

   nrw=file_head.nrw;
   nfct=file_head.nfct;
   nsrc=file_head.nsrc;
   shift_lnrw();
   set_rw();

   for (irw=0;irw<nrw;irw++)
   {
      printf("Reweighting factor no %d:\n",irw);
      if (nfct[irw]>1)
         printf("Factorized into %d factors.\n",nfct[irw]);
      if (nsrc[irw]>1)
         printf("Using %d random source fields.\n\n",nsrc[irw]);
      else
         printf("Using 1 random source field.\n\n");

      if (nms>=100)
         print_auto(nms,rw[irw]);
      else
         print_jack(1,nms,rw+irw,f);

      printf("\n");
   }

   if (nrw>1)
   {
      printf("Product of all reweighting factors:\n\n");
      set_rwall();

      if (nms>=100)
         print_auto(nms,rwall);
      else
         print_jack(1,nms,&rwall,f);

      printf("\n");
   }

   print_plot(argv[1]);
   exit(0);
}
