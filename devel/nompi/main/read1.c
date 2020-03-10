
/*******************************************************************************
*
* File read1.c
*
* Copyright (C) 2010-2014, 2018, 2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reads and evaluates data from the <name>.dat files created by the programs
* qcd1, qcd2, ym1 and ym2.
*
* The program writes the history of the MD energy deficit dH, the acceptance
* flag iac and the average plaquette to the file <name>.run.dat in the plots
* directory. In addition, some information about the distribution of dH and
* the integrated autocorrelation time of the plaquette are printed to stdout.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "extras.h"

typedef struct
{
   int nt,iac;
   double dH,avpl;
} dat_t;

static int nms,neff,first,last,step;
static dat_t *adat;


static int read_dat(int n,dat_t *ndat,FILE *fin)
{
   int i,ir,ic,endian;
   stdint_t istd[2];
   double dstd[2];

   endian=endianness();
   ic=0;

   for (i=0;i<n;i++)
   {
      ir=fread(istd,sizeof(stdint_t),2,fin);
      ir+=fread(dstd,sizeof(double),2,fin);

      if (ir!=4)
         return ic;

      if (endian==BIG_ENDIAN)
      {
         bswap_int(2,istd);
         bswap_double(2,dstd);
      }

      (*ndat).nt=(int)(istd[0]);
      (*ndat).iac=(int)(istd[1]);

      (*ndat).dH=dstd[0];
      (*ndat).avpl=dstd[1];

      ic+=1;
      ndat+=1;
   }

   return ic;
}


static void read_file(char *fin)
{
   int n,nt,nl,dn,ie;
   long ipos;
   dat_t ndat;
   FILE *fdat;

   fdat=fopen(fin,"rb");
   error(fdat==NULL,1,"read_file [read1.c]","Unable to open data file");

   printf("Read data from file %s\n\n",fin);
   ipos=ftell(fdat);
   nms=0;

   while (read_dat(1,&ndat,fdat)==1)
      nms+=1;

   error(nms==0,1,"read_file [read1.c]",
         "Empty data file");

   adat=malloc(nms*sizeof(*adat));
   error(adat==NULL,1,"read_file [read1.c]",
         "Unable to allocate data array");

   fseek(fdat,ipos,SEEK_SET);
   error(read_dat(nms,adat,fdat)!=nms,1,"read_file [read1.c]",
         "Error while reading data file");
   fclose(fdat);

   ie=0;
   dn=1;
   nl=adat[0].nt;

   for (n=1;n<nms;n++)
   {
      nt=adat[n].nt;

      if (n==1)
         dn=nt-nl;
      else
         ie|=(nt!=(nl+dn));

      nl=nt;
   }

   error(ie!=0,1,"read_file [read1.c]","Varying trajectory separation");
}


static void select_range(void)
{
   int fst,lst,stp;

   fst=adat[0].nt;
   lst=adat[nms-1].nt;

   if (nms>1)
   {
      stp=adat[1].nt-adat[0].nt;
      printf("There are %d measurements (trajectories no %d - %d by %d).\n",
             nms,fst,lst,stp);
   }
   else
   {
      stp=1;
      printf("There is 1 measurement (trajectory no %d).\n",fst);
   }

   printf("Range first,last,step of trajectories to analyse: ");
   scanf("%d",&first);
   scanf(",");
   scanf("%d",&last);
   scanf(",");
   scanf("%d",&step);

   error((step<=0)||((step%stp)!=0),1,"select_range [read1.c]",
         "Step must be positive and divisible by the trajectory separation");

   if (first<fst)
   {
      first=first+((fst-first)/step)*step;
      if (first<fst)
         first+=step;
   }

   error((first>lst)||(((first-fst)%stp)!=0)||(last<first),1,
         "select_cnfg_range [read1.c]","Improper trajectory range");

   if (last>lst)
      last=lst;
   last=last-(last-first)%step;
   neff=(last-first)/step+1;
   error(neff<2,1,"select_range [read1.c]",
         "Selected range contains less than 2 measurements");

   printf("Keep %d measurements (trajectories no %d - %d by %d).\n\n",
          neff,first,last,step);
}


static double tail(int n,double *a,double amx)
{
   int i,c;

   c=0;

   for (i=0;i<n;i++)
   {
      if (a[i]>amx)
         c+=1;
   }

   return (double)(c)/(double)(n);
}


static double f(int nx,double x[])
{
   return x[0];
}


static void print_plot(char *fin)
{
   int n,ims,nt;
   char base[NAME_SIZE],plt_file[NAME_SIZE],*p;
   FILE *fout;

   p=strstr(fin,".dat");
   error(p==NULL,1,"print_plot [read1.c]","Unexpected data file name");
   n=p-fin;

   p=strrchr(fin,'/');
   if (p==NULL)
      p=fin;
   else
      p+=1;
   n-=(p-fin);

   error(n>=NAME_SIZE,1,"print_plot [read1.c]","File name is too long");
   strncpy(base,p,n);
   base[n]='\0';

   error(name_size("plots/%s.run.dat",base)>=NAME_SIZE,1,
         "print_plot [read1.c]","File name is too long");
   sprintf(plt_file,"plots/%s.run.dat",base);
   fout=fopen(plt_file,"w");
   error(fout==NULL,1,"print_plot [read1.c]",
         "Unable to open output file");

   fprintf(fout,"#\n");
   fprintf(fout,"# Data written by the program qcd1, qcd2, ym1 or ym2\n");
   fprintf(fout,"# --------------------------------------------------\n");
   fprintf(fout,"#\n");
   fprintf(fout,"# Number of measurements = %d\n",neff);
   fprintf(fout,"#\n");
   fprintf(fout,"# nt:   trajectory number\n");
   fprintf(fout,"# dH:   MD energy deficit\n");
   fprintf(fout,"# iac:  acceptance flag\n");
   fprintf(fout,"#\n");
   fprintf(fout,"#  nt         dH      iac    <tr{U(p)}>\n");
   fprintf(fout,"#\n");

   for (ims=0;ims<nms;ims++)
   {
      nt=adat[ims].nt;

      if ((nt>=first)&&(nt<=last)&&(((nt-first)%step)==0))
      {
         fprintf(fout," %5d  ",nt);
         fprintf(fout," % .4e  ",adat[ims].dH);
         fprintf(fout," %1d  ",adat[ims].iac);
         fprintf(fout," %.8e",adat[ims].avpl);
         fprintf(fout,"\n");
      }
   }

   fclose(fout);

   printf("Data printed to file %s\n\n",plt_file);
}


int main(int argc,char *argv[])
{
   int ims,n,nt;
   double *a,abar;

   error(argc!=2,1,"main [read1.c]","Syntax: read1 <filename>");

   printf("\n");
   printf("Simulation of QCD (program qcd1, qcd2, ym1 or ym2)\n");
   printf("--------------------------------------------------\n\n");

   read_file(argv[1]);
   select_range();

   a=malloc(neff*sizeof(*a));
   error(a==NULL,1,"main [read1.c]",
         "Unable to allocate data array");

   for (ims=0,n=0;ims<nms;ims++)
   {
      nt=adat[ims].nt;

      if ((nt>=first)&&(nt<=last)&&(((nt-first)%step)==0))
      {
         a[n]=fabs(adat[ims].dH);
         n+=1;
      }
   }

   printf("Fraction of trajectories with |dH| larger than\n\n");
   printf("    1.0: %.4f\n",tail(neff,a,1.0));
   printf("    2.0: %.4f\n",tail(neff,a,2.0));
   printf("   10.0: %.4f\n",tail(neff,a,10.0));
   printf("  100.0: %.4f\n",tail(neff,a,100.0));
   printf(" 1000.0: %.4f\n",tail(neff,a,1000.0));
   printf("\n");

   for (ims=0,n=0;ims<nms;ims++)
   {
      nt=adat[ims].nt;

      if ((nt>=first)&&(nt<=last)&&(((nt-first)%step)==0))
      {
         a[n]=exp(-adat[ims].dH);
         n+=1;
      }
   }

   printf("<exp(-dH)> = %.5f (%.5f)\n",
          average(neff,a),sigma0(neff,a));

   for (ims=0,n=0;ims<nms;ims++)
   {
      nt=adat[ims].nt;

      if ((nt>=first)&&(nt<=last)&&(((nt-first)%step)==0))
      {
         if (adat[ims].dH>0.0)
            a[n]=exp(-adat[ims].dH);
         else
            a[n]=1.0;

         n+=1;
      }
   }

   printf("<min{1,exp(-dH)}> = %.5f (%.5f)\n",
          average(neff,a),sigma0(neff,a));

   for (ims=0,n=0;ims<nms;ims++)
   {
      nt=adat[ims].nt;

      if ((nt>=first)&&(nt<=last)&&(((nt-first)%step)==0))
      {
         a[n]=(double)(adat[ims].iac);
         n+=1;
      }
   }

   printf("<iac> = %.3f (%.3f)\n\n",
          average(neff,a),sigma0(neff,a));

   for (ims=0,n=0;ims<nms;ims++)
   {
      nt=adat[ims].nt;

      if ((nt>=first)&&(nt<=last)&&(((nt-first)%step)==0))
      {
         a[n]=adat[ims].avpl;
         n+=1;
      }
   }

   printf("The integrated autocorrelation time and the associated"
          "\nstatistical error sigma of the plaquette is estimated ");

   if (neff>100)
      printf("using the\nnumerically determined "
             "autocorrelation function.\n\n");
   else
      printf("by binning the\ndata and by calculating "
             "the jackknife errors of the binned series.\n\n");

   printf("The autocorrelation times are given in numbers of measurements\n"
          "separated by %d trajectories.\n\n",step);

   if (neff>=100)
      abar=print_auto(neff,a);
   else
      abar=print_jack(1,neff,&a,f);

   printf(" <tr{U(p)}> = %1.9f\n\n",abar);

   print_plot(argv[1]);
   exit(0);
}
