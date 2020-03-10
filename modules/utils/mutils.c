
/*******************************************************************************
*
* File mutils.c
*
* Copyright (C) 2005-2016, 2018, 2019 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Utility functions used in main programs.
*
*   void check_machine(void)
*     Checks the endianess of the machine, the basic data sizes and the
*     IEEE 754 compliance of the floating-point arithmetic on process 0.
*
*   int find_opt(int argc,char *argv[],char *opt)
*     On process 0, this program compares the string opt with the arguments
*     argv[1],..,argv[argc-1] and returns the position of the first argument
*     that matches the string. If there is no matching argument, or if the
*     program is called from another process, the return value is 0.
*
*   int fdigits(double x)
*     Returns the smallest integer n such that the value of x printed with
*     print format %.nf coincides with x up to a relative error at most a
*     few times the machine precision DBL_EPSILON.
*
*   void check_dir(char* dir)
*     This program checks whether the directory "dir" is accessible and
*     aborts the main program with an informative error message if this
*     is not the case. The program may be called on any subset of MPI
*     processes with varying arguments.
*
*   void check_dir_root(char* dir)
*     On process 0, this program checks whether the directory "dir" is
*     accessible and aborts the main program with an informative error
*     message if this is not the case. When called on other processes,
*     the program does nothing.
*
*   int check_file(char* file,char* mode)
*     Returns 1 if the file "file" can be opened with mode "mode" and
*     0 otherwise. The program may be called locally and does not perform
*     any communications.
*
*   int name_size(char *format,...)
*     Returns the length of the string that would be printed by calling
*     sprintf(*,format,...). The format string can be any combination of
*     literal text and the conversion specifiers %s, %d and %.nf (where
*     1<=n<=32). These must correspond to the string, integer and double
*     arguments after the format string. The lengths of the strings must
*     be less than NAME_SIZE. An error occurs if any of these conditions
*     is violated or if the calculated length is larger than INT_MAX.
*
*   long find_section(char *title)
*     On process 0, this program scans stdin for a line starting with
*     the string "[title]" (after any number of blanks). It terminates
*     with an error message if no such line is found or if there are
*     several of them. The program returns the offset of the line from
*     the beginning of the file and positions the file pointer to the
*     next line. On processes other than 0, the program does nothing
*     and returns -1L.
*
*   long read_line(char *tag,char *format,...)
*     On process 0, this program reads a line of text and data from stdin
*     in a controlled manner, as described in the notes below. The tag can
*     be the empty string "" and must otherwise be an alpha-numeric word
*     that starts with a letter. If it is not empty, the program searches
*     for the tag in the current section. An error occurs if the tag is not
*     found. The program returns the offset of the line from the beginning
*     of the file and positions the file pointer to the next line. On
*     processes other than 0, the program does nothing and returns -1L.
*
*   int count_tokens(char *tag)
*     On process 0, this program finds and reads a line from stdin, exactly
*     as read_line(tag,..) does, and returns the number of tokens found on
*     that line after the tag. Tokens are separated by white space (blanks,
*     tabs or newline characters) and comments (text beginning with #) are
*     ignored. On exit, the file pointer is positioned at the next line. If
*     called on other processes, the program does nothing and returns 0.
*
*   void read_iprms(char *tag,int n,int *iprms)
*     On process 0, this program finds and reads a line from stdin, exactly
*     as read_line(tag,..) does, reads n integer values from that line after
*     the tag and assigns them to the elements of the array iprms. An error
*     occurs if less than n values are found on the line. The values must be
*     separated by white space (blanks, tabs or newline characters). On exit,
*     the file pointer is positioned at the next line. When called on other
*     processes, the program does nothing.
*
*   void read_dprms(char *tag,int n,double *dprms)
*     On process 0, this program finds and reads a line from stdin, exactly
*     as read_line(tag,..) does, reads n double values from that line after
*     the tag and assigns them to the elements of the array dprms. An error
*     occurs if less than n values are found on the line. The values must be
*     separated by white space (blanks, tabs or newline characters). On exit,
*     the file pointer is positioned at the next line. When called on other
*     processes, the program does nothing.
*
*   void copy_file(char *in,char *out)
*     Copies the file "in" to the file "out" in binary mode. An error occurs
*     if the file copy is not successful.
*
* The programs in this module do not involve any communications and can be
* called locally.
*
* See the notes doc/qsum.pdf for possible explanations and remedies if
* check_machine() complains about the floating-point arithmetic being not
* IEEE 754 compliant.
*
* The programs find_section() and read_line() serve to read structured
* input parameter files (such as the *.in in the directory main; see
* main/README.infiles).
*
* Parameter lines that can be read by read_line() must be of the form
*
*   tag v1 v2 ...
*
* where v1,v2,... are data values (strings, integers or floating-point
* numbers) separated by blanks. If the tag is empty, the first data value
* may not be a string. Such lines are read by calling
*
*   read_line(tag,format,&var1,&var2,...)
*
* where var1,var2,... are the variables to which the values v1,v2,... are
* to be assigned. The format string must include the associated sequence
* of conversion specifiers %s, %d, %f or %lf without any modifiers. Other
* tokens are not allowed in the format string, except for additional blanks
* and a newline character at the end of the string (none of these have any
* effect).
*
* The programs find_section() and read_line() ignore blank lines and any text
* appearing after the character #. Lines longer than NAME_SIZE-1 characters are
* not permitted. Each section may occur at most once and, within each section,
* a line tag may not appear more than once. The number of characters written
* to the target string variables is at most NAME_SIZE-1. Buffer overflows are
* thus excluded if the target strings are of size NAME_SIZE or larger.
*
*******************************************************************************/

#define MUTILS_C

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "global.h"

static int nsmx=0;
static char text[512];
static char line[NAME_SIZE+1];
static char *numstr;


void check_machine(void)
{
   int my_rank,np,ie;
   double l0,l1,l2,l3;
   double x,y1,y2,y3;
   double q1[2],q2[2],q3[2];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   MPI_Comm_size(MPI_COMM_WORLD,&np);

   if (my_rank==0)
   {
      error_root(np!=NPROC,1,"check_machine [mutils.c]",
                 "Actual number of MPI processes does not match NPROC");
      error_root(endianness()==UNKNOWN_ENDIAN,1,"check_machine [mutils.c]",
                 "Unknown endianness");
      error_root((FLT_RADIX!=2)||(FLT_MANT_DIG!=24)||(FLT_MIN_EXP!=-125)||
                 (FLT_MAX_EXP!=128)||((double)(FLT_EPSILON)!=ldexp(1.0,-23)),
                 1,"check_machine [mutils.c]",
                 "'float' data format is not IEEE 754 compliant");
      error_root((DBL_MANT_DIG!=53)||(DBL_MIN_EXP!=-1021)||
                 (DBL_MAX_EXP!=1024)||((double)(DBL_EPSILON)!=ldexp(1.0,-52)),
                 1,"check_machine [mutils.c]",
                 "'double' data format is not IEEE 754 compliant");
      error_root((sizeof(stdint_t)!=4)||(sizeof(float)!=4)||
                 (sizeof(double)!=8)||(sizeof(qflt)!=16),1,
                 "check_machine [mutils.c]","Unexpected basic data sizes");
      error_root((sizeof(su3_dble)!=144)||(sizeof(su3_alg_dble)!=64)||
                 (sizeof(spinor_dble)!=192),1,"check_machine [mutils.c]",
                 "Field element structures are not properly packed");

      x=1.0;
      y1=0.5*DBL_EPSILON;
      y2=y1*(1.0+DBL_EPSILON);
      y3=y1*(1.0-0.5*DBL_EPSILON);
      y1=(x+y1)-x;
      y2=(x+y2)-x;
      y3=(x+y3)-x;

      ie=(y1!=0.0);
      ie|=(y2!=DBL_EPSILON);
      ie|=(y3!=0.0);

      q1[0]=123456789012345.0;
      q1[1]=0.0012345;
      q2[0]=ldexp(1.0,-8)*q1[0];
      q2[1]=0.0;

      add_qflt(q1,q2,q3);
      acc_qflt(q2[0],q1);
      q2[0]=-q2[0];
      add_qflt(q2,q3,q3);
      acc_qflt(q2[0],q1);

      ie|=(q1[0]!=q3[0]);
      ie|=(q1[1]!=q3[1]);

      q3[0]-=123456789012345.0;
      q3[1]-=0.0012345;

      ie|=(q3[0]!=0.0);
      ie|=(q3[1]!=-ldexp(1.0,-61));

      error_root(ie,1,"check_machine [mutils.c]",
                 "Floating-point arithmetic is not IEEE 754 compliant\n"
                 "or a non-default rounding is used");

      l0=(double)(L0+1);
      l1=(double)(L1+1);
      l2=(double)(L2+1);
      l3=(double)(L3+1);
      ie|=((4.0*l0*l1*l2*l3)>(double)(INT_MAX));

      error_root(ie,1,"check_machine [mutils.c]",
                 "Local lattice size is out of range");
   }
}


int find_opt(int argc,char *argv[],char *opt)
{
   int my_rank,k;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      for (k=1;k<argc;k++)
         if (strcmp(argv[k],opt)==0)
            return k;
   }

   return 0;
}


int fdigits(double x)
{
   int m,n,ne,k;
   double y,z;

   if (x==0.0)
      return 0;

   y=fabs(x);
   z=DBL_EPSILON*y;
   m=floor(log10(y+z));
   n=0;
   ne=1;

   for (k=0;k<(DBL_DIG-m);k++)
   {
      z=sqrt((double)(ne))*DBL_EPSILON*y;

      if (((y-floor(y))<=z)||((ceil(y)-y)<=z))
         break;

      y*=10.0;
      ne+=1;
      n+=1;
   }

   return n;
}


void check_dir(char* dir)
{
   int my_rank,nc,n;
   char *tmp_file;
   FILE *tmp;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   nc=strlen(dir);
   tmp_file=malloc((nc+7+3*sizeof(int))*sizeof(char));
   error_loc(tmp_file==NULL,1,"check_dir [mutils.c]",
             "Unable to allocate name string");
   sprintf(tmp_file,"%s/.tmp_%d",dir,my_rank);

   n=0;
   tmp=fopen(tmp_file,"rb");

   if (tmp==NULL)
   {
      n=1;
      tmp=fopen(tmp_file,"wb");
   }

   nc=sprintf(text,"Unable to access directory ");
   strncpy(text+nc,dir,512-nc);
   text[511]='\0';
   error_loc(tmp==NULL,1,"check_dir [mutils.c]",text);
   fclose(tmp);

   if (n==1)
      remove(tmp_file);
   free(tmp_file);
}


void check_dir_root(char* dir)
{
   int my_rank,nc,n;
   char *tmp_file;
   FILE *tmp;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      nc=strlen(dir);
      tmp_file=malloc((nc+6)*sizeof(char));
      error_root(tmp_file==NULL,1,"check_dir_root [mutils.c]",
                 "Unable to allocate name string");
      sprintf(tmp_file,"%s/.tmp",dir);

      n=0;
      tmp=fopen(tmp_file,"rb");

      if (tmp==NULL)
      {
         n=1;
         tmp=fopen(tmp_file,"wb");
      }

      error_root(tmp==NULL,1,"check_dir_root [mutils.c]",
                 "Unable to access directory %s from process 0",dir);

      fclose(tmp);
      if (n==1)
         remove(tmp_file);
      free(tmp_file);
   }
}


int check_file(char* file,char* mode)
{
   FILE *tmp;

   tmp=fopen(file,mode);

   if (tmp==NULL)
      return 0;
   else
   {
      fclose(tmp);
      return 1;
   }
}


static void alloc_numstr(int ns)
{
   if (ns>nsmx)
   {
      if (nsmx>0)
         free(numstr);

      numstr=malloc(ns*sizeof(char));
      error_loc(numstr==NULL,1,"alloc_numstr [mutils.c]",
                "Unable to allocate auxiliary string");
      nsmx=ns;
   }
}


static size_t num_size(int n,double r)
{
   int ns;
   double a;

   ns=n+5;
   a=fabs(r);

   if (a>=10.0)
      ns+=(int)(log10(a));

   alloc_numstr(ns);
   sprintf(numstr,"%.*f",n,r);

   return strlen(numstr);
}


int name_size(char *format,...)
{
   int n,ie;
   size_t nf,nl,ns,nmx;
   char *pc;
   va_list args;

   alloc_numstr(3*sizeof(int)+4);

   va_start(args,format);
   pc=format;
   nmx=(size_t)(INT_MAX);
   nf=strlen(format);
   nl=0;
   ie=0;

   while ((strchr(pc,'%')!=NULL)&&(ie==0))
   {
      pc=strchr(pc,'%')+1;

      if (pc[0]=='s')
      {
         strncpy(line,va_arg(args,char*),NAME_SIZE);
         line[NAME_SIZE]='\0';
         ns=strlen(line);

         if (ns<NAME_SIZE)
         {
            nf-=2;

            if ((nmx-ns)>=nl)
               nl+=ns;
            else
               ie=1;
         }
         else
            ie=1;
      }
      else if (pc[0]=='d')
      {
         sprintf(numstr,"%d",va_arg(args,int));
         ns=strlen(numstr);
         nf-=2;

         if ((nmx-ns)>=nl)
            nl+=ns;
         else
            ie=1;
      }
      else if (pc[0]=='.')
      {
         if (sscanf(pc,".%d",&n)==1)
         {
            sprintf(numstr,".%df",n);

            if ((n>=1)&&(n<=32)&&(pc==strstr(pc,numstr)))
            {
               ns=num_size(n,va_arg(args,double));

               if (n<10)
                  nf-=4;
               else
                  nf-=5;

               if ((nmx-ns)>=nl)
                  nl+=ns;
               else
                  ie=1;
            }
            else
               ie=1;
         }
         else
            ie=1;
      }
      else
         ie=1;
   }

   va_end(args);

   if ((ie==0)&&((nmx-nf)>=nl))
      return (int)(nl+nf);
   else
   {
      error_loc(1,1,"name_size [mutils.c]",
                "Improper format string or string data");
      return INT_MAX;
   }
}


static int cmp_text(char *text1,char *text2)
{
   size_t n1,n2;
   char *p1,*p2;

   p1=text1;
   p2=text2;

   while (1)
   {
      p1+=strspn(p1," \t\n");
      p2+=strspn(p2," \t\n");
      n1=strcspn(p1," \t\n");
      n2=strcspn(p2," \t\n");

      if (n1!=n2)
         return 0;
      if (n1==0)
         return 1;
      if (strncmp(p1,p2,n1)!=0)
         return 0;

      p1+=n1;
      p2+=n1;
   }
}


static char *get_line(void)
{
   char *s,*c;

   s=fgets(line,NAME_SIZE+1,stdin);

   if (s!=NULL)
   {
      error_root(strlen(line)==NAME_SIZE,1,"get_line [mutils.c]",
                 "Input line is longer than NAME_SIZE-1");

      c=strchr(line,'#');
      if (c!=NULL)
         c[0]='\0';
   }

   return s;
}


long find_section(char *title)
{
   int my_rank,ie;
   long ofs,sofs;
   char *s,*pl,*pr;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      rewind(stdin);
      sofs=-1L;
      ofs=ftell(stdin);
      s=get_line();

      while (s!=NULL)
      {
         pl=strchr(line,'[');
         pr=strchr(line,']');

         if ((pl==(line+strspn(line," \t")))&&(pr>pl))
         {
            pl+=1;
            pr[0]='\0';

            if (cmp_text(pl,title)==1)
            {
               error_root(sofs>=0L,1,"find_section [mutils.c]",
                          "Section [%s] occurs more than once",title);
               sofs=ofs;
            }
         }

         ofs=ftell(stdin);
         s=get_line();
      }

      error_root(sofs==-1L,1,"find_section [mutils.c]",
                 "Section [%s] not found",title);
      ie=fseek(stdin,sofs,SEEK_SET);
      error_root(ie!=0,1,"find_section [mutils.c]",
                 "Unable to go to section [%s]",title);
      get_line();

      return sofs;
   }
   else
      return -1L;
}


static void check_tag(char *tag)
{
   if (tag[0]=='\0')
      return;

   error_root((strspn(tag," 0123456789.")!=0L)||
              (strcspn(tag," \n")!=strlen(tag)),1,
              "check_tag [mutils.c]","Improper tag %s",tag);
}


static long find_tag(char *tag)
{
   int ie;
   long tofs,lofs,ofs;
   char *s,*pl,*pr;

   ie=0;
   tofs=-1L;
   lofs=ftell(stdin);
   rewind(stdin);
   ofs=ftell(stdin);
   s=get_line();

   while (s!=NULL)
   {
      pl=strchr(line,'[');
      pr=strchr(line,']');

      if ((pl==(line+strspn(line," \t")))&&(pr>pl))
      {
         if (ofs<lofs)
         {
            ie=0;
            tofs=-1L;
         }
         else
            break;
      }
      else
      {
         pl=line+strspn(line," \t");
         pr=pl+strcspn(pl," \t\n");
         pr[0]='\0';

         if (strcmp(pl,tag)==0)
         {
            if (tofs!=-1L)
               ie=1;
            tofs=ofs;
         }
      }

      ofs=ftell(stdin);
      s=get_line();
   }

   error_root(tofs==-1L,1,"find_tag [mutils.c]","Tag %s not found",tag);
   error_root(ie!=0,1,"find_tag [mutils.c]",
              "Tag %s occurs more than once in the current section",tag);

   ie=fseek(stdin,tofs,SEEK_SET);
   error_root(ie!=0,1,"find_tag [mutils.c]",
              "Unable to go to line with tag %s",tag);

   return tofs;
}


long read_line(char *tag,char *format,...)
{
   int my_rank,is,ic;
   long tofs;
   char *pl,*p;
   va_list args;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      check_tag(tag);

      if (tag[0]!='\0')
      {
         tofs=find_tag(tag);
         get_line();
         pl=line+strspn(line," \t");
         pl+=strcspn(pl," \t\n");
      }
      else
      {
         p=format;
         p+=strspn(p," ");
         error_root(strstr(p,"%s")==p,1,"read_line [mutils.c]",
                    "String data after empty tag");
         tofs=ftell(stdin);
         pl=get_line();
      }

      va_start(args,format);

      for (p=format;;)
      {
         p+=strspn(p," ");
         ic=0;
         is=2;

         if ((p[0]=='\0')||(p[0]=='\n'))
            break;
         else if (p==strstr(p,"%s"))
            ic=sscanf(pl,"%s",va_arg(args,char*));
         else if (p==strstr(p,"%d"))
            ic=sscanf(pl,"%d",va_arg(args,int*));
         else if (p==strstr(p,"%f"))
            ic=sscanf(pl,"%f",va_arg(args,float*));
         else if (p==strstr(p,"%lf"))
         {
            is=3;
            ic=sscanf(pl,"%lf",va_arg(args,double*));
         }
         else
            error_root(1,1,"read_line [mutils.c]",
                       "Incorrect format string %s on line with tag %s",
                       format,tag);

         error_root(ic!=1,1,"read_line [mutils.c]",
                    "Missing data item(s) on line with tag %s",tag);

         p+=is;
         pl+=strspn(pl," \t");
         pl+=strcspn(pl," \t\n");
      }

      va_end(args);

      return tofs;
   }
   else
      return -1L;
}


int count_tokens(char *tag)
{
   int my_rank,n;
   char *s;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      check_tag(tag);

      if (tag[0]!='\0')
      {
         find_tag(tag);
         s=get_line();
         s+=strspn(s," \t");
         s+=strcspn(s," \t\n");
      }
      else
         s=get_line();

      s+=strspn(s," \t\n");
      n=0;

      while (s[0]!='\0')
      {
         n+=1;
         s+=strcspn(s," \t\n");
         s+=strspn(s," \t\n");
      }

      return n;
   }
   else
      return 0;
}


void read_iprms(char *tag,int n,int *iprms)
{
   int my_rank,nc,ic,i;
   char *s;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      check_tag(tag);

      if (tag[0]!='\0')
      {
         find_tag(tag);
         s=get_line();
         s+=strspn(s," \t");
         s+=strcspn(s," \t\n");
      }
      else
         s=get_line();

      s+=strspn(s," \t\n");
      nc=0;

      while ((s[0]!='\0')&&(nc<n))
      {
         ic=sscanf(s,"%d",&i);

         if (ic==1)
         {
            iprms[nc]=i;
            nc+=1;
            s+=strcspn(s," \t\n");
            s+=strspn(s," \t\n");
         }
         else
            break;
      }

      error_root(nc!=n,1,"read_iprms [mutils.c]","Incorrect read count");
   }
}


void read_dprms(char *tag,int n,double *dprms)
{
   int my_rank,nc,ic;
   double d;
   char *s;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      check_tag(tag);

      if (tag[0]!='\0')
      {
         find_tag(tag);
         s=get_line();
         s+=strspn(s," \t");
         s+=strcspn(s," \t\n");
      }
      else
         s=get_line();

      s+=strspn(s," \t\n");
      nc=0;

      while ((s[0]!='\0')&&(nc<n))
      {
         ic=sscanf(s,"%lf",&d);

         if (ic==1)
         {
            dprms[nc]=d;
            nc+=1;
            s+=strcspn(s," \t\n");
            s+=strspn(s," \t\n");
         }
         else
            break;
      }

      error_root(nc!=n,1,"read_dprms [mutils.c]","Incorrect read count");
   }
}


void copy_file(char *in,char *out)
{
   int c;
   FILE *fin,*fout;

   fin=fopen(in,"rb");
   error_loc(fin==NULL,1,"copy_file [mutils.c]","Unable to open input file");

   fout=fopen(out,"wb");
   error_loc(fout==NULL,1,"copy_file [mutils.c]","Unable to open output file");

   c=getc(fin);

   while (feof(fin)==0)
   {
      putc(c,fout);
      c=getc(fin);
   }

   if ((ferror(fin)==0)&&(ferror(fout)==0))
   {
      fclose(fin);
      fclose(fout);
   }
   else
      error_loc(1,1,"copy_file [mutils.c]","Read or write error");
}
