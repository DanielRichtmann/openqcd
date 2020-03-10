
/*******************************************************************************
*
* File mutils.c
*
* Copyright (C) 2005-2013, 2018 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Utility functions used in main programs.
*
*   void check_machine(void)
*     Checks the endianess of the machine, the basic data sizes and the
*     IEEE 754 compliance of the floating-point representation.
*
*   int find_opt(int argc,char *argv[],char *opt)
*     On process 0, this program compares the string opt with the arguments
*     argv[1],..,argv[argc-1] and returns the position of the first argument
*     that matches the string. If there is no matching argument, or if the
*     program is called from another process, the return value is 0.
*
*   int digits(double x,double dx,char *fmt)
*     Assuming x is a value with error dx, this program returns the number n
*     of fractional digits to print so that all significant digits plus two
*     more are shown. The print format fmt has to be "e" or "f" depending on
*     whether the number is to be printed using the "%.ne" or "%.nf" format
*     string. In the second case dx has to be in the range 0<dx<1, and
*     (int)(10^n*dx) is then a two-digit integer that represents the error
*     in the last two digits of the printed value.
*
*   int fdigits(double x)
*     Returns the smallest integer n such that the value of x printed with
*     print format %.nf coincides with x up to a relative error at most a
*     few times the machine precision DBL_EPSILON.
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
*   long find_section(FILE *stream,char *title)
*     This program scans stream for a line starting with the string "[title]"
*     (after any number of blanks). It terminates with an error message if no
*     such line is found or if there are several of them. The program returns
*     the offset of the line from the beginning of the file and positions the
*     file pointer to the next line.
*
*   long read_line(FILE *stream,char *tag,char *format,...)
*     This program scans stream and reads a line of text in a controlled
*     manner, as described in the notes below. The tag can be the empty
*     string "" and must otherwise be an alpha-numeric word that starts
*     with a letter. If it is not empty, the program searches for the tag
*     in the current section. An error occurs if the tag is not found. The
*     program returns the offset of the line from the beginning of the file
*     and positions the file pointer to the next line.
*
*   int count_tokens(FILE *stream,char *tag)
*     This program finds and reads a line from stream, exactly as read_line()
*     does, and returns the number of tokens found on that line after the tag.
*     Tokens are separated by white space (blanks, tabs or newline characters)
*     and comments (text beginning with #) are ignored. On exit, the file
*     pointer is positioned at the next line.
*
*   void read_iprms(FILE *stream,char *tag,int n,int *iprms)
*     This program finds and reads a line from stream, exactly as read_line()
*     does, reads n integer values from that line after the tag and assigns
*     them to the elements of the array iprms. An error occurs if less than
*     n values are found on the line. The values must be separated by white
*     space (blanks, tabs or newline characters). On exit, the file pointer
*     is positioned at the next line.
*
*   void read_dprms(FILE *stream,char *tag,int n,double *dprms)
*     This program finds and reads a line from stream, exactly as read_line()
*     does, reads n double values from that line after the tag and assigns
*     them to the elements of the array iprms. An error occurs if less than
*     n values are found on the line. The values must be separated by white
*     space (blanks, tabs or newline characters). On exit, the file pointer
*     is positioned at the next line.
*
* The program check_machine() aborts the main program with the error message
* "Floating-point arithmetic is not IEEE 754 compliant" if the machine makes
* use of extended precision registers. Machines with x86 instruction set in
* the 32 bit mode do this by default. Some special compiler options may be
* required in these cases to enforce strict IEEE 754 compliance.
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
#include "su3.h"
#include "utils.h"

static int nsmx=0;
static char line[NAME_SIZE+1];
static char *numstr;


void check_machine(void)
{
   int ie;
   double x,y1,y2,y3;
   double q1[2],q2[2],q3[2];

   error(endianness()==UNKNOWN_ENDIAN,1,"check_machine [mutils.c]",
         "Unknown endianness");
   error((FLT_RADIX!=2)||(FLT_MANT_DIG!=24)||(FLT_MIN_EXP!=-125)||
         (FLT_MAX_EXP!=128)||((double)(FLT_EPSILON)!=ldexp(1.0,-23)),
         1,"check_machine [mutils.c]",
         "'float' data format is not IEEE 754 compliant");
   error((DBL_MANT_DIG!=53)||(DBL_MIN_EXP!=-1021)||
         (DBL_MAX_EXP!=1024)||((double)(DBL_EPSILON)!=ldexp(1.0,-52)),
         1,"check_machine [mutils.c]",
         "'double' data format is not IEEE 754 compliant");
   error((sizeof(stdint_t)!=4)||(sizeof(float)!=4)||
         (sizeof(double)!=8),1,"check_machine [mutils.c]",
         "Unexpected basic data sizes");

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

   error(ie,1,"check_machine [mutils.c]",
         "Floating-point arithmetic is not IEEE 754 compliant\n"
         "or a non-default rounding is used");
}


int find_opt(int argc,char *argv[],char *opt)
{
   int k;

   for (k=1;k<argc;k++)
   {
      if (strcmp(argv[k],opt)==0)
         return k;
   }

   return 0;
}


int digits(double x,double dx,char *fmt)
{
   error(dx<0.0,1,"digits [mutils.c]","Improper input data (negative error)");

   if (strcmp(fmt,"e")==0)
   {
      if (dx==0.0)
         return DBL_DIG;
      else if (dx>=fabs(x))
         return 1;
      else
         return (int)(floor(1.0+log10(fabs(x)))-floor(log10(dx)));
   }
   else if (strcmp(fmt,"f")==0)
   {
      error((dx==0.0)||(dx>=1.0),1,"digits [mutils.c]",
            "Improper input data (error out of range for fixed format)");

      return (int)(1.0-floor(log10(dx)));
   }
   else
      error(1,1,"digits [mutils.c]","Unknown data format");

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


static char *get_line(FILE *stream)
{
   char *s,*c;

   s=fgets(line,NAME_SIZE+1,stream);

   if (s!=NULL)
   {
      error(strlen(line)==NAME_SIZE,1,"get_line [mutils.c]",
                 "Input line is longer than NAME_SIZE-1");

      c=strchr(line,'#');
      if (c!=NULL)
         c[0]='\0';
   }

   return s;
}


long find_section(FILE *stream,char *title)
{
   int ie;
   long ofs,sofs;
   char *s,*pl,*pr;


   rewind(stream);
   sofs=-1L;
   ofs=ftell(stream);
   s=get_line(stream);

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
            error(sofs>=0L,1,"find_section [mutils.c]",
                       "Section [%s] occurs more than once",title);
            sofs=ofs;
         }
      }

      ofs=ftell(stream);
      s=get_line(stream);
   }

   error(sofs==-1L,1,"find_section [mutils.c]",
              "Section [%s] not found",title);
   ie=fseek(stream,sofs,SEEK_SET);
   error(ie!=0,1,"find_section [mutils.c]",
              "Unable to go to section [%s]",title);
   get_line(stream);

   return sofs;
}


static void check_tag(char *tag)
{
   if (tag[0]=='\0')
      return;

   error((strspn(tag," 0123456789.")!=0L)||
              (strcspn(tag," \n")!=strlen(tag)),1,
              "check_tag [mutils.c]","Improper tag %s",tag);
}


static long find_tag(FILE *stream,char *tag)
{
   int ie;
   long tofs,lofs,ofs;
   char *s,*pl,*pr;

   ie=0;
   tofs=-1L;
   lofs=ftell(stream);
   rewind(stream);
   ofs=ftell(stream);
   s=get_line(stream);

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

      ofs=ftell(stream);
      s=get_line(stream);
   }

   error(tofs==-1L,1,"find_tag [mutils.c]","Tag %s not found",tag);
   error(ie!=0,1,"find_tag [mutils.c]",
              "Tag %s occurs more than once in the current section",tag);

   ie=fseek(stream,tofs,SEEK_SET);
   error(ie!=0,1,"find_tag [mutils.c]",
              "Unable to go to line with tag %s",tag);

   return tofs;
}


long read_line(FILE *stream,char *tag,char *format,...)
{
   int is,ic;
   long tofs;
   char *pl,*p;
   va_list args;

   check_tag(tag);

   if (tag[0]!='\0')
   {
      tofs=find_tag(stream,tag);
      get_line(stream);
      pl=line+strspn(line," \t");
      pl+=strcspn(pl," \t\n");
   }
   else
   {
      p=format;
      p+=strspn(p," ");
      error(strstr(p,"%s")==p,1,"read_line [mutils.c]",
                 "String data after empty tag");
      tofs=ftell(stream);
      pl=get_line(stream);
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
         error(1,1,"read_line [mutils.c]",
                    "Incorrect format string %s on line with tag %s",
                    format,tag);

      error(ic!=1,1,"read_line [mutils.c]",
                 "Missing data item(s) on line with tag %s",tag);

      p+=is;
      pl+=strspn(pl," \t");
      pl+=strcspn(pl," \t\n");
   }

   va_end(args);

   return tofs;
}


int count_tokens(FILE *stream,char *tag)
{
   int n;
   char *s;


   check_tag(tag);

   if (tag[0]!='\0')
   {
      find_tag(stream,tag);
      s=get_line(stream);
      s+=strspn(s," \t");
      s+=strcspn(s," \t\n");
   }
   else
      s=get_line(stream);

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


void read_iprms(FILE *stream,char *tag,int n,int *iprms)
{
   int nc,ic,i;
   char *s;

   check_tag(tag);

   if (tag[0]!='\0')
   {
      find_tag(stream,tag);
      s=get_line(stream);
      s+=strspn(s," \t");
      s+=strcspn(s," \t\n");
   }
   else
      s=get_line(stream);

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

   error(nc!=n,1,"read_iprms [mutils.c]","Incorrect read count");
}


void read_dprms(FILE *stream,char *tag,int n,double *dprms)
{
   int nc,ic;
   double d;
   char *s;

   check_tag(tag);

   if (tag[0]!='\0')
   {
      find_tag(stream,tag);
      s=get_line(stream);
      s+=strspn(s," \t");
      s+=strcspn(s," \t\n");
   }
   else
      s=get_line(stream);

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

   error(nc!=n,1,"read_dprms [mutils.c]","Incorrect read count");
}
