set style data dots
set nokey
set xrange [0:11.18988]
set yrange [  6.00302 : 49.29832]
set arrow from  1.25332,   6.00302 to  1.25332,  49.29832 nohead
set arrow from  2.50665,   6.00302 to  2.50665,  49.29832 nohead
set arrow from  4.27911,   6.00302 to  4.27911,  49.29832 nohead
set arrow from  5.15633,   6.00302 to  5.15633,  49.29832 nohead
set arrow from  6.40965,   6.00302 to  6.40965,  49.29832 nohead
set arrow from  7.66297,   6.00302 to  7.66297,  49.29832 nohead
set arrow from  9.43544,   6.00302 to  9.43544,  49.29832 nohead
set arrow from 10.31266,   6.00302 to 10.31266,  49.29832 nohead
set xtics ("G"  0.00000,"X"  1.25332,"M"  2.50665,"G"  4.27911,"Z"  5.15633,"R"  6.40965,"A"  7.66297,"Z|X"  9.43544,"R|M" 10.31266,"A" 11.18988)
 plot "pwscf_band.dat"
