pois      = 0.48;                                           %Poisson Ratio
young     = 8000e-12;                                       %Young's Modulus (N/micron^2)
pixelsize = 0.3263;                                          %Pixel/micron ratio: 0.1030 for 60x
                                                            %                    0.1634 for 40x
                                                            %                    0.3263 for 20x
                                                            %                    0.6515 for 10x
%subcellnum

switch whichcell;


case 1;    % 
   first   		= [200002];   								% Reference image (tryp)
   second  		= [200001]; 								% b, h5-1nd, col
   
   
   j1 			= 80; 	% Left   200 232 
   i1 			= 80; 	% Top
   j2 			= 350;	% Right   1159 1191
   i2 			= 350; 	% Bottom                            %(i2-i1+1) must be divisible by blocksize
                            
                            %(j2-j1+1) must be divisible by blocksize
   subcellnum   = 1;
   i2=ceil((i2-i1+1)/32)*32+i1-1;
   j2=ceil((j2-j1+1)/32)*32+j1-1;

   
   %% This part makes the image square. In case the dimentions exceed the image size, comment this section.
   if (i2-i1)>(j2-j1)
       j2=j1+(i2-i1);
   else
       i2=i1+(j2-j1);
   end;
   
end;

% i11=i1; i22=i2;
% i1=j1; i2=j2;
% j1=i11; j2=i22;


ba = second(1);
