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

It is somewhat embarassing that it took us 4 years to open source this work due
to our laziness. :(
