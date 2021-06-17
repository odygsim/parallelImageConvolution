#!/bin/csh

mpicc -g -o mpi_conv mpi_image_conv.c -lm

#set imagename = r4_waterfall_grey_1920_2520.raw
set execname = mpi_conv
set inumrows = 2520
set numcols = 1920
set maxiters = 30
#set conviters = ( 30 2 )
set imagetype = rgb
set machfile = machines
set inum = 0

# mpicc -o conv mpi_image_conv.c -L/usr/local/mpip3/lib -lmpiP -lm -lbfd -liberty 
# /// analoga allaxe pou exei egkatastathei to mpiP
foreach imagename (r2_waterfall_1920_2520.raw r4_waterfall_1920_2520.raw)
   echo $imagename

   @ inum = $inum + 1

   if ($inum == 1) then 
	@ numrows = $inumrows / 2
        #echo $numrows
   else if ($inum == 2) then
	@ numrows = $inumrows / 4
        #echo $numrows
   endif

    foreach conviters (30 2)
        echo $imagename $conviters
	foreach nprocs (1 2 4 6 8 9 16 25 32 36 64 72 100 108)
	    foreach cntr  (1 2 3)
        	  echo "np = " $nprocs " - iteration : " $cntr 
	          mpiexec -f $machfile -np $nprocs $execname $imagename $numrows $numcols $maxiters $conviters $imagetype
		  #echo $cntr	
    	    end
	end
   end
end
# sed '1i P5\n1920 2520\n255' conv_waterfall_grey_1920_2520.raw > conv_waterfall_grey_1920_2520.pgm

