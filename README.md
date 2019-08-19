__1. Fuction:__  
*  Copy data from cpu to PHI (initialization on server1).  
*  Copy data from PHI to cpu on server1. 
*  Communicate data from cpu on server 1 to cpu on server 2. 
*  Copy data from cpu to gpu on server 2. 
https://github.com/XianMeng/phi-to-gpu/blob/master/Function-phi-to-gpu.jpg

__2. Compile:__    
nvcc -ccbin=mpiicc cpu.cu -o measuretime -lmicmgmt

__3. Run:__     
* On seerver02:  
`mpirun -hostfile hostfile ./measeuretime`   
It is ok to run on server02    
The screen print is as follows:
https://github.com/XianMeng/phi-to-gpu/blob/master/phi-to-gpu-screen-out.png

