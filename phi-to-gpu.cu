/* 
*  mpi.c: main program 
*  Aug. 2019
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "/usr/include/miclib.h"
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "/usr/include/miclib.h"

#define ping 101
#define data_Size 100000000
//#define DEBUG

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)
#define RETAIN free_if(0)

void rank1(int * cpu1_data, int dataSize);

//Time variables
double GPU_to_CPU_time;
double CPU_to_GPU_time;
double PHI_to_CPU_time;
double timep;

//Recording time
struct timeval start, end , startPHI , endPHI , startGTC , endGTC, start111, end111;
double tstart , tend , tstartPHI , tendPHI , tstartGTC , tendGTC, tstart111, tend111;

//Data size variables
double GB, GBps_GPUtoCPU, GBps_mpi, GBps_CPUtoGPU, GBps_CPUtoPHI ;

//Data in cpu0
//int *cpu_to_gpu_init_data = NULL;
//int *data_in_cpu_from_gpu = NULL;
//int *send = NULL;

//Data in cpu1
int *data_in_cpu1;
int *phi_data;

int main(int argc, char* argv[]) {
    //Data in cpu0
    int *cpu_to_gpu_init_data = NULL;
    int *data_in_cpu_from_gpu = NULL;
    int *send = NULL;
    
    //Data in cpu1
   // int *data_in_cpu1;
   // int *phi_data;

    char ProcessName[MPI_MAX_PROCESSOR_NAME]; 

    //For recordiing time
    double start, finish, mpi_time;
    int  Length; 

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get MPI number and rank; 
    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    //Get process name
    MPI_Get_processor_name(ProcessName, &Length);

    //Initialization: Rank 0 sent data to GPU
    if(commRank == 0)  { 
        printf("Running using %d MPIs in total.\n",commSize);
        printf("I am rank %d\t ProcessName %s \t\n", commRank, ProcessName);
   
        send = (int *)malloc(sizeof(int)*data_Size);
          if(send==NULL) {
             fprintf(stderr,"Could not get %d bytes of send\n",send);
             exit(1);
          }

    phi_data = (int *)malloc(sizeof(int)*data_Size);

    // Initialize host data
    int i;
    for(i = 0; i < data_Size; i++) {
        phi_data[i] = 4;
    }

    #pragma offload_transfer target(mic) \
      in( phi_data : length(data_Size) ALLOC )
    
   //Copy data from host to phi
   #pragma offload_transfer target(mic) in( phi_data : length(data_Size)  REUSE )

   //Copy data from phi to host
   gettimeofday(&startPHI, NULL);
   #pragma offload_transfer target(mic) out( phi_data : length(data_Size)  REUSE )
   gettimeofday(&endPHI, NULL);

   tstartPHI = startPHI.tv_sec + startPHI.tv_usec/1000000.;
   tendPHI = endPHI.tv_sec + endPHI.tv_usec/1000000.;
   PHI_to_CPU_time = (tendPHI - tstartPHI);

   #ifdef DEBUG
   printf("PHI_to_CPU_time : %f \n", PHI_to_CPU_time);
   printf("############################ PHI to CPU copy time  ##############################\n");
   printf(" %d int data,      time is:%f seconds\n",data_Size, PHI_to_CPU_time);
   printf("#####################################################################################\n");
    #endif

   //Buffer for MPI send
    for ( int j = 0; j < data_Size; j++ )
      {
        send[j]=phi_data[i];
      }
       
    //Send data from cpu in MPI 0 to cpu in MPI 1
    start = MPI_Wtime();
    MPI_Send(send, data_Size, MPI_INT, 1, ping,MPI_COMM_WORLD);
    finish = MPI_Wtime();

    mpi_time = finish - start;

    #ifdef DEBUG
    printf("############################ CPU0 to CPU1 transfer time ############################\n");
    printf(" %d int data,      time is:%f seconds\n",  data_Size, mpi_time);
    printf("#####################################################################################\n");
    #endif
 
    //Send GPU_to_CPU_time and mpi_time to MPI 1, because it does not know.
    MPI_Send(&mpi_time, 1, MPI_DOUBLE, 1, 300, MPI_COMM_WORLD);
    MPI_Send(&PHI_to_CPU_time, 1, MPI_DOUBLE, 1, 400, MPI_COMM_WORLD);

     //Deallocate PHI memory
     #pragma offload_transfer target(mic)   in( phi_data : length(data_Size) FREE  )
    }
    else if (commRank == 1) {
        printf("I am rank %d\t ProcessName %s \t\n", commRank, ProcessName);

        data_in_cpu1 = (int *)malloc(sizeof(int)*data_Size);
        if(data_in_cpu1==NULL) {
          fprintf(stderr,"Could not get %d bytes of memory data_in_cpu1\n",data_Size);
          exit(1);
        }

        
        //Receive data from MPI 0
        MPI_Recv(data_in_cpu1, data_Size, MPI_INT, 0, ping, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   
             
        //Receive GPU_to_CPU_time and mpi_time from MPI 0
        double mpi_time_recv;
        double PHI_to_CPU_time_recv;
        MPI_Recv(&mpi_time_recv, 1, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&PHI_to_CPU_time_recv, 1, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 
        
        //send data in cpu 1 to gpu1
        rank1(data_in_cpu1,data_Size);

        // data_Size size of (int) first turn bits into bytes and then devide by 1000000000 turn it into GB
        GB =data_Size*sizeof(int)/(1000.0*1000.0*1000.0);
        //GBps_GPUtoCPU = GB/GPU_to_CPU_time_recv ;
        GBps_mpi = GB/mpi_time_recv ;
        GBps_CPUtoGPU = GB/CPU_to_GPU_time ;
        GBps_CPUtoPHI = GB/PHI_to_CPU_time_recv ;

        //Print out the GPU0 to PHI1 transfer time on the screen
        printf("########################################   MIC (server1) to GPU (server2) total transfer time  #######################################\n");
        printf("%d int data, size = %0.4lf MB, PHI (server1) to CPU (server1) transfer time = %f,  transfer bandwidth = %0.2lf GB/s\n", data_Size, GB*1000.0, PHI_to_CPU_time_recv,GBps_CPUtoPHI );
        printf("%d int data, size = %0.4lf MB, CPU (server1) to CPU (server2) transfer time = %f, transfer bandwidth = %0.2lf GB/s \n", data_Size, GB*1000.0, mpi_time_recv, GBps_mpi );
        printf("%d int data, size = %0.4lf MB, CPU (server2) to GPU (server2) transfer time = %f, transfer bandwidth = %0.2lf GB/s \n", data_Size, GB*1000.0, CPU_to_GPU_time, GBps_CPUtoGPU);
        printf("Total time is the sum of the above time. \n");
        printf(" %d int data,      total  time  is:%f seconds\n", data_Size, PHI_to_CPU_time_recv + CPU_to_GPU_time + mpi_time_recv );
        printf("######################################################################################################################################\n");
    }
   
    // Cleanup
    free(cpu_to_gpu_init_data);
    free(data_in_cpu_from_gpu);
    MPI_Finalize();
    if(commRank == 0) {
      printf("Test PASSED\n");
    }

    return 0;
}

void rank1(int * cpu1_data, int dataSize){
   //Variables for recoding time
   struct timeval start, end;
   double tstart , tend;

   //data in cpu;
   int * data_in_cpu;

   //Allocate GPU memory
   int * gpu_data_from_cpu = NULL; //data from cpu to gpu
   cudaMalloc((void**)&gpu_data_from_cpu, dataSize * sizeof(int));

   // Allocate pinned host memory
   cudaHostAlloc(&data_in_cpu, sizeof(int) * dataSize, cudaHostAllocDefault);

   int i;
   for (i=1;i<dataSize;i++)
     {
     data_in_cpu[i]=cpu1_data[i];
     }

   //Copy data from cpu to gpu
   gettimeofday(&start, NULL);
   cudaMemcpy(gpu_data_from_cpu, data_in_cpu, sizeof(int) * dataSize, cudaMemcpyHostToDevice); 
   gettimeofday(&end, NULL);

   tstart = start.tv_sec + start.tv_usec/1000000.;
   tend = end.tv_sec + end.tv_usec/1000000.;
   CPU_to_GPU_time = (tend - tstart);

   #ifdef DEBUG
   printf("############################ CPU1 to GPU1 transfer time  ##############################\n");
   printf(" %d int data,      time is:%f seconds\n",dataSize, CPU_to_GPU_time);
   printf("#####################################################################################\n");
   #endif

   //Free GPU memory
   cudaFree(gpu_data_from_cpu);

   //Free host memory
   cudaFreeHost(data_in_cpu);
   }
