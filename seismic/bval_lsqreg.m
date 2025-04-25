function [b, a, std_err] = bval_lsqreg(mag, bin)
      %mag=Mag; bin=0.1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Written by AVN 
% standard least squares regression fit
%
% Last modified on  Fri Jul 27 11:45:11 EDT 2007
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mmax=max(mag);
mmin=min(mag);
i=0;
sx=0;sy=0;sxy=0;sxx=0;
for count=mmin:bin:mmax
	i=i+1;
   x(i)=count; 
   ylin=length(mag(mag >= count));
   if ylin>0 
   	y(i)=log10(ylin);
   else
	y(i)=0;
   end
   sx=sx+x(i);
   sy=sy+y(i);
   sxy=sxy+x(i)*y(i);
   sxx=sxx+x(i)^2;
end
nbins=i;
i=0;
num=0;denom=0;
for count=mmin:bin:mmax
     i=i+1;
         num=num+(x(i)-sx/nbins)*(y(i)-sy/nbins);
         denom=denom+(x(i)-sx/nbins)^2.;
end
   b=-num/denom;
   a=sy/nbins+sx/nbins*b;

   std_err=0;
end
