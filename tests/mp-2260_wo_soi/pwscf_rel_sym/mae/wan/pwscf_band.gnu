set style data dots
set nokey
set xrange [0:10.35727]
set yrange [  6.48056 : 48.57950]
set arrow from  1.15155,   6.48056 to  1.15155,  48.57950 nohead
set arrow from  2.30310,   6.48056 to  2.30310,  48.57950 nohead
set arrow from  3.93164,   6.48056 to  3.93164,  48.57950 nohead
set arrow from  4.76297,   6.48056 to  4.76297,  48.57950 nohead
set arrow from  5.91452,   6.48056 to  5.91452,  48.57950 nohead
set arrow from  7.06607,   6.48056 to  7.06607,  48.57950 nohead
set arrow from  8.69461,   6.48056 to  8.69461,  48.57950 nohead
set arrow from  9.52594,   6.48056 to  9.52594,  48.57950 nohead
set xtics ("G"  0.00000,"X"  1.15155,"M"  2.30310,"G"  3.93164,"Z"  4.76297,"R"  5.91452,"A"  7.06607,"Z|X"  8.69461,"R|M"  9.52594,"A" 10.35727)
 plot "pwscf_band.dat"
