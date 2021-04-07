% clc;
% clear all;
% close all;

% N = 100;
% M = 3;
% build the scale free network by the BA model

function Ba=ScaleFree(N,M)

for i = 1 : M
   Ba( i ).Sec = [ 1 : i - 1 , i + 1 : M ];
   Ba( i ).D = M - 1;
end

T = M * ( M - 1 );

for i = M : N - 1
   for num = 1 : i
      P( num ) = Ba( num ).D / T;
   end
   r = rand;
   s = 0;
   for num = 1 : i
      s = s + P( num );
      if r < s
         break;
      end
   end
   Ba( num ).Sec = [ Ba( num ).Sec,i + 1 ];
   Ba( num ).D = Ba( num ).D + 1;
   Ba( i + 1 ).Sec = [ num ];
   Ba( i + 1 ).D = 1;
   T = T + 2;
end
end