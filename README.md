# SOI_FFT
Segment-of-interest low-communication FFT algorithm.
Instead of 3 all-to-all communications required for multi-node 1D FFT using
Cooley-Tukey decomposition, SOI FFT only needs 1 all-to-all.
The all-to-all in SOI FFT has a bit more communication volume than each
all-to-all of Cooley-Tukey FFT depending on oversampling factor, so
communication volume saving will be 2.x .

A Framework for Low-Communication 1-D FFT, Ping Tak Peter Tang, Jongsoo Park,
Daehyun Kim, and Vladimir Petrov, The International Conference for High
Performance Computing, Networking, Storage, and Analysis, 2012,
http://pcl.intel-research.net/publications/framework-2012.pdf

Tera-Scale 1D FFT with Low-Communication Algorithm and Intel Xeon Phi
Coprocessors, Jongsoo Park, Ganesh Bikshandi, Karthikeyan Vaidyanathan, Ping
Tak Peter Tang, Pradeep Dubey, and Daehyun Kim, The International Conference for
High Performance Computing, Networking, Storage, and Analysis, 2013,
http://pcl.intel-research.net/publications/a34-park.pdf

![Alt text](results/soi_xeon_phi.png?raw=true "Performance Comparison of SOI and Cooley-Tukey FFT on Xeon and Xeon Phi")

It is somewhat embarrassing that it took us 4 years to open source this work due
to our laziness. :(
Still, we hope our code can be useful for those who need a fast multi-node FFT
or someone who wants to play with ideas of further improving SOI FFT.
For example, what would be the most efficient way of using SOI FFT for
multi-dimensional FFT is an interesting problem.
We also looked at compressing data sent during all-to-all that can be useful
when the output has banded structure (most signal power concentrated to a
narrow frequency range), but was not able to get significant speedups for
real-life examples. See use_vlc option for our attempt.
