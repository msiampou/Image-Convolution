/*
compilation - openmp: mpicc -fopenmp imgconv.c -lm
run: mpiexec -n <processes> ./a.out width height type loops
*/

/*
compilation - mpi: mpicc imgconv.c -lm
run: mpiexec -f <machines> -n <processes> ./a.out width height type loops
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "omp.h"
#include <time.h>
#include <limits.h>
#include <stdint.h>

#define filename1 "../img/waterfall_grey_1920_2520.raw"
#define filename2 "../img/waterfall_1920_2520.raw"

typedef struct subarray{
    int cols;
    int rows;
    int _div;
} coords;

typedef struct destination{
    int west;
    int east;
    int north;
    int south;
} dest_processes;

typedef struct covonlution_data {
    MPI_Datatype row_datatype;
    MPI_Datatype col_datatype;
    int rows;
    int cols;
    int bytes;
    dest_processes p;
    uint8_t *t0;
    uint8_t *t1;
} convolution_data_t;

MPI_Status __status;

#define CHECK_SIMILARITY 0

void perror_exit(const char *message){
    perror(message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

int check_similarity(uint8_t *t0, uint8_t *t1, int pos){
    if(t0[pos]!=t1[pos])
        return 1;
    else
        return 0;
}

int convolute(uint8_t *t0, uint8_t *t1, int srow, int scol, int lrow, 
                int lcol, int cols, int bytes, float h[3][3], int flag){

    int check = 0;

#   pragma omp parallel for num_threads(1) collapse(2) schedule(static)

    for (int i = srow; i < lrow +1; i++){
        for (int j = scol; j < lcol +1; j++){
            float val = 0, bval = 0, gval = 0;
            int x=0;
            for (int k = i-1; k <= i+1; k++){
                int y=0;
		        for (int l = bytes*j - bytes; l <= bytes*j + bytes ; l += bytes){
		            val += t0[bytes*(cols+2) * k + l] * h[x][y];
		            if(bytes==3){
		    	        gval += t0[bytes*(cols+2) * k + l+1] * h[x][y];
			            bval += t0[bytes*(cols+2) * k + l+2] * h[x][y];
		            }
		            y++;
		        }
                x++;
	        }
            t1[bytes*(cols+2) * i + bytes*j] = val;

            /* We check for similarity under 2 circumstances:
               1. When inner data convolution takes place
               2. No differences between bytes have been found so far
            */
            if ((flag) && (!check) && (CHECK_SIMILARITY)) {
                check = check_similarity(t0,t1,(bytes*(cols+2) * i + bytes*j));
            }
            if(bytes==3){
		        t1[bytes*(cols+2) * i + bytes*j+1] = gval;
		        t1[bytes*(cols+2) * i + bytes*j+2] = bval;
                if ((flag) && (!check) && (CHECK_SIMILARITY)){
                    check = check_similarity(t0,t1,(bytes*(cols+2) * i + bytes*j+1));
                    check = check_similarity(t0,t1,(bytes*(cols+2) * i + bytes*j+2));
                }
	        }
	    }
    }
    return check;
}

int convolution(convolution_data_t cd){
    MPI_Datatype row_datatype = cd.row_datatype;
    MPI_Datatype col_datatype = cd.col_datatype;
    int rows = cd.rows;
    int cols = cd.cols;
    int bytes = cd.bytes;
    dest_processes p = cd.p;
    uint8_t *t0 = cd.t0;
    uint8_t *t1 = cd.t1;

    int flag;

    /*gaussian blur*/
    float h[3][3] = {{1/16.0, 2/16.0, 1/16.0}, {2/16.0, 4/16.0, 2/16.0}, {1/16.0, 2/16.0, 1/16.0}};

    MPI_Request send_req[4], recv_req[4];

    MPI_Isend(&t0[bytes*((cols+2) +1)], 1, row_datatype, p.north, 0, MPI_COMM_WORLD, &send_req[0]);
    MPI_Irecv(&t0[bytes], 1, row_datatype, p.north, 0, MPI_COMM_WORLD, &recv_req[0]);

    MPI_Isend(&t0[bytes*((cols+2) +1)], 1, col_datatype, p.west, 0, MPI_COMM_WORLD, &send_req[1]);
    MPI_Irecv(&t0[bytes*(cols+2)], 1, col_datatype, p.west, 0, MPI_COMM_WORLD, &recv_req[1]);

    MPI_Isend(&t0[bytes*((rows*(cols+2)) +1)], 1, row_datatype, p.south, 0, MPI_COMM_WORLD, &send_req[2]);
    MPI_Irecv(&t0[bytes*(((rows+1)*(cols+2)) +1)], 1, row_datatype, p.south, 0, MPI_COMM_WORLD, &recv_req[2]);

    MPI_Isend(&t0[bytes*((cols+2) +cols)], 1, col_datatype, p.east, 0, MPI_COMM_WORLD, &send_req[3]);
    MPI_Irecv(&t0[bytes*((cols+2) +(cols+1))], 1, col_datatype, p.east, 0, MPI_COMM_WORLD, &recv_req[3]);


    /* compute inner data */
    flag = 1;
    int check = convolute(t0,t1,1,1,rows,cols,cols,bytes,h,flag);
    flag = 0;

    /* compute outer data */
    if (p.north != MPI_PROC_NULL){
        MPI_Wait(&recv_req[0], &__status);
        convolute(t0,t1,1,2,1,cols-1,cols,bytes,h,flag);
    }

    if (p.west != MPI_PROC_NULL){
        MPI_Wait(&recv_req[1], &__status);
        convolute(t0,t1,2,1,rows-1,1,cols,bytes,h,flag);
    }

    if (p.south != MPI_PROC_NULL){
        MPI_Wait(&recv_req[2], &__status);
        convolute(t0,t1,rows,2,rows,cols-1,cols,bytes,h,flag);
    }

    if (p.east != MPI_PROC_NULL){
        MPI_Wait(&recv_req[3], &__status);
        convolute(t0,t1,2,cols,rows-1,cols,cols,bytes,h,flag);
    }

    /* Deal with neighbors */
    if ((p.north != MPI_PROC_NULL) && (p.west != MPI_PROC_NULL))
        convolute(t0,t1,1,1,1,1,cols,bytes,h,flag);
    if ((p.south != MPI_PROC_NULL) && (p.west != MPI_PROC_NULL))
        convolute(t0,t1,rows,1,rows,1,cols,bytes,h,flag);
    if ((p.north != MPI_PROC_NULL) && (p.east != MPI_PROC_NULL))
        convolute(t0,t1,cols,cols,cols,rows,cols,bytes,h,flag);
    if ((p.south != MPI_PROC_NULL) && (p.east != MPI_PROC_NULL))
        convolute(t0,t1,rows,cols,rows,cols,cols,bytes,h,flag);

    /* wait for all processes to finish */
    if (p.north != MPI_PROC_NULL)  MPI_Wait(&send_req[0], &__status);
    if (p.west != MPI_PROC_NULL)   MPI_Wait(&send_req[1], &__status);
    if (p.south != MPI_PROC_NULL)  MPI_Wait(&send_req[2], &__status);
    if (p.east != MPI_PROC_NULL)   MPI_Wait(&send_req[3], &__status);

    return check;
}

void compute_neighbors(int my_rank, int num_processes, int _div, dest_processes* p){

    p->north = my_rank - _div;
    p->south = my_rank + _div;
    p->west = my_rank - 1;
    p->east = my_rank + 1;

    if (p->north < 0)               p->north = MPI_PROC_NULL;
    if (p->south >= num_processes)  p->south = MPI_PROC_NULL;
    if (p->west < 0)                p->west = MPI_PROC_NULL;
    if (p->east >= num_processes)   p->east = MPI_PROC_NULL;
}

uint8_t* parallel_read(int my_rank, int _div, int rows_per_proc, int cols_per_proc, int bytes, int width){

    /* calculating starting column and row for each process */
    int start_row = (my_rank / _div) * rows_per_proc;
    int start_col = (my_rank % _div) * cols_per_proc;

    uint8_t *t0 = calloc((rows_per_proc+2) * (bytes*(cols_per_proc+2)), sizeof(uint8_t));
    if(t0 == NULL)
        perror_exit("Sorry, cannot allocate enough memory");

    MPI_File fh;

    if(bytes == 1)
        MPI_File_open(MPI_COMM_WORLD, filename1, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    else
        MPI_File_open(MPI_COMM_WORLD, filename2, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    for(int i=1; i<=rows_per_proc; i++){
        MPI_File_seek(fh, (bytes*(start_row + i-1) * width + bytes*start_col), MPI_SEEK_SET);
        MPI_File_read(fh, &t0[((bytes*(cols_per_proc+2))*i + bytes)], bytes*cols_per_proc, MPI_BYTE, &__status);
    }

    MPI_File_close(&fh);
    return t0;
}

void parallel_write(int my_rank, int _div, int rows_per_proc, int cols_per_proc, int bytes, int width, uint8_t* t1){

    /* calculating starting column and row for each process */
    int start_row = (my_rank / _div) * rows_per_proc;
    int start_col = (my_rank % _div) * cols_per_proc;

    char* out = strdup("blur_image.raw");
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, out, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    for(int i=1; i<=rows_per_proc; i++){
        MPI_File_seek(fh, (bytes*(start_row + i-1) * width + bytes*start_col), MPI_SEEK_SET);
        MPI_File_write(fh, &t1[((bytes*(cols_per_proc+2)*i) + bytes)], bytes*cols_per_proc, MPI_BYTE, &__status);
    }

    MPI_File_close(&fh);
    free(out);
}

/* Searching the best number to devide height and width for each process */
void array_div(int num_processes, coords** subarray, int width, int height){
    (*subarray)->_div = sqrt((double)num_processes);
    (*subarray)->rows = height / (*subarray)->_div;
    (*subarray)->cols = width / (*subarray)->_div;
}

int main(int argc, char **argv){

    int my_rank, comm_sz;
    int rows_per_proc, cols_per_proc, _div, bytes, loops, width, height;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* if I am master process */
    if(my_rank == 0){

        coords* subarray = (coords*)malloc(sizeof(coords));

        width = atoi(argv[1]);
        height = atoi(argv[2]);

        array_div(comm_sz,&subarray,width,height);

        if (!strcmp((argv[3]),"grey"))
            bytes = 1;
        else
            bytes = 3;
        loops = atoi((argv[4]));

        rows_per_proc = subarray->rows;
        cols_per_proc = subarray->cols;
        _div = subarray->_div;
    }

    /* broadcast parameters to other processes */
    MPI_Bcast(&rows_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols_per_proc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&loops, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* t0: image before convolution */
    uint8_t *t0 = parallel_read(my_rank,_div,rows_per_proc,cols_per_proc,bytes,width);

    /* t1: image after convolution */
    uint8_t *t1 = calloc((rows_per_proc+2)*(bytes*(cols_per_proc+2)), sizeof(uint8_t));
    if(t1 == NULL)
        perror_exit("Sorry, cannot allocate enough memory");

    MPI_Datatype column_datatype;
    MPI_Datatype row_datatype;

    MPI_Type_vector(rows_per_proc, bytes, bytes*(cols_per_proc+2), MPI_BYTE, &column_datatype);
    MPI_Type_contiguous(bytes*cols_per_proc, MPI_BYTE, &row_datatype);
    MPI_Type_commit(&column_datatype);
    MPI_Type_commit(&row_datatype);

    dest_processes p;
    compute_neighbors(my_rank,comm_sz,_div,&p);

    MPI_Barrier(MPI_COMM_WORLD);

    /*The actual loop*/
    /*Convoluting image a number of times*/
    double global_timer = 0.0, local_timer = MPI_Wtime();
    convolution_data_t cd = {
        row_datatype,
        column_datatype,
        rows_per_proc,
        cols_per_proc,
        bytes,
        p,
        t0,
        t1
    };
    for(int i=0; i<loops; i++){
        int local_flag, global_sum;
        local_flag = convolution(cd);

        /*Checking for similarity between src and dest image*/
        if (CHECK_SIMILARITY){
            MPI_Allreduce(&local_flag, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if(global_sum == 0){
                if(my_rank == 0)
                    printf("Convolution is not producing a new image. Breaking the loop ...\n");
                break;
            }
        }

        /*Swap arrays*/
        uint8_t *tmp = t0;
        t0  = t1;
        t1 = tmp;
    }
    local_timer = MPI_Wtime() - local_timer;

    /*Finding the slowest process*/
    MPI_Reduce(&local_timer, &global_timer, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(my_rank == 0)
        printf("Elapsed Time: %.4lf\n", global_timer);

    parallel_write(my_rank,_div,rows_per_proc,cols_per_proc,bytes,width,t0);

    free(t0);
    free(t1);

    MPI_Type_free(&row_datatype);
    MPI_Type_free(&column_datatype);

    MPI_Finalize();
    return 0;
}
