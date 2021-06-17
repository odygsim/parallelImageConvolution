/* File:     mpi_image_conv.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

typedef enum {IMG_GREY, IMG_RGB} img_color;
typedef enum {NN, NS, NW, NE, NNW, NNE, NSW, NSE} neighbor_orientation;

/* Calculate the number of columns splits */
int columns_splits(int trows, int tcols, int nprocs);
/* Get and check command line parameters */
void get_parameters(int argc, char** argv, char** imgName, int* numRows, int* numCols,
		int* maxIters, int* convIters, img_color* imgType);
/* Memory allocation to image arrays */
int allocate_img_arrays(int locRows, int locCols, int imgType, uint8_t** imgcurr, uint8_t** imgnext);
/* Locate neighbor processes */
void establish_neighbors(int pid, int totalRows, int totalCols,
		int locRows, int locCols, int initRow, int initCol, int colSplits,
		int* north, int* south, int* west, int* east,
		int* nwest, int* swest, int* neast, int* seast);
/* Calculate convolution for a range of pixel rows-columns - returns if something has changed */
int calculate_convolution(uint8_t* imgcurr, uint8_t* imgnext, int startRow, int endRow, int startCol,
		int endCol, int locCols, int imgType, float filter[3][3], int thread_count);
/* Copying edge/corner pixels to frame when there is no neighbor*/
void copy_edges_to_frame(char* imgnext, int imgType, int cntr_min, int cntr_max,
		int cntr_mult, int stdpart_l, int stdpart_r);
/* Getting the initial addresses to be looped for copying edges/corners */
void getting_addreses_parts_for_copying_edges(neighbor_orientation edge, int imgType, int locRows, int locCols,
		int* cntr_min, int* cntr_max, int* cntr_mult, int* stdpart_l, int* stdpart_r);
/* Parallel read of part of image to processes */
void read_image_parallel(MPI_Comm comm, int procId, char* imgName, int imgType, uint8_t* img,
		int totalRows, int totalCols, int rowSplits, int colSplits, MPI_Datatype* filetype, MPI_Datatype* memtype);
/* Write in parallel convoluted image to file with name "conv_" & old_name */
void write_image_parallel(MPI_Comm comm, char* imgName, uint8_t* img, MPI_Datatype* filetype, MPI_Datatype* memtype);
/* Reference global image frame ghost pixels to the values of edges pixels (avoids copying)
void frame_pixels_referencing(int North, int South, int West, int East, int NorthWest, int NorthEast,
		int SouthWest, int SouthEast, int imgType, uint8_t* locImg, int locRows, int locCols);*/
/* Calc the position of and element at (row,col) pixel in the enhanced local array (including halo area) */
int arraypos(int row, int col , int locRows, int locCols, int halo);

/***** NOT USED *****
 * Parallel reading of full image parts to processes  */
void read_full_image(MPI_Comm comm, int pid, char* imgName, int imgType, char* img,
		int initRow, int initCol, int locRows, int locCols, int totalRows, int totalCols,
		MPI_Datatype grey_rowtype, MPI_Datatype grey_coltype, MPI_Datatype rgb_rowtype,
		MPI_Datatype rgb_coltype);
/***** NOT USED *****
 * Parallel writing of convoluted image - not fully parallel */
void write_full_image(MPI_Comm comm, int pid, char* previmgName, int imgType, char* img,
		int initRow, int initCol, int locRows, int locCols, int totalRows, int totalCols,
		MPI_Datatype grey_rowtype, MPI_Datatype grey_coltype, MPI_Datatype rgb_rowtype,
		MPI_Datatype rgb_coltype);
/****** NOT USED ******
 * Initialize pointers of neighbors rows-columns-pixels in image array for send-receive actions */
void initialize_neighbors_pointers_in_image_array(char* imgcurr, int locRows, int locCols,int imgType,
		int North, int South, int West, int East, int NorthWest, int NorthEast, int SouthWest, int SouthEast,
		char** nptr_S, char** nptr_R, char** sptr_S, char** sptr_R, char** wptr_S, char** wptr_R,
		char** eptr_S, char** eptr_R, char** swptr_S, char** swptr_R, char** septr_S, char** septr_R,
		char** nwptr_S, char** nwptr_R, char** neptr_S, char** neptr_R);
/****** NOT USED *****
 * Find the pointer position in the image array */
char* positionptr(char* img, int cRow, int cCol, int cols);
/* Print statistics of running to std file */
void print_log_to_file(int nProcs, char* imgName, int nrows, int ncols, int maxiters, int conviters,
		double readtime, double runtime, double writetime, int nthreads);
/* Get system time for statistics */
struct tm* mycurrtime();

/*--------------------------------------------------------------------------------------------------------*/
int main(int argc, char** argv) {

	char* imgName;			// File name of the image
	int imgNmLen;			// File name char length
	int totalRows;			// Total number of image rows
	int totalCols;			// Total number of image columns
	int maxIters;			// Max number of convolution calculations to be performed
	int convIters;			// Number of iterations to check for convergence (no image change)
	int rowSplits;			// Number of splits for rows
	int colSplits;			// Number of splits for columns
	int locRows;			// Local (process) number of image rows
	int locCols;			// Local (process) number of image columns
	int nProcs;				// Number of processes
	int i,k;
	int thread_count=4;

	img_color imgType;		// Type of image
	neighbor_orientation neigOrient;	// To be used for

	// Initialize MPI and get number of processes and the specific process id
	MPI_Init(&argc, &argv);
	int procId;				// Id of specific process
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procId);

	int North = -1;			// North neighbor process id
	int South = -1;			// South neighbor process id
	int West = -1;			// West neighbor process id
	int East = -1;			// East neighbor process id
	int NorthWest = -1;		// NorthWest neighbor process id
	int NorthEast = -1;		// NorthEast neighbor process id
	int SouthWest = -1;		// SouthWest neighbor process id
	int SouthEast = -1;		// SouthEast neighbor process id

	/* Setting MPI Datatypes for row and column vectors and corner elements
	 * (in order to be generic)*/
	MPI_Datatype grey_rowtype;
	MPI_Datatype grey_coltype;
	MPI_Datatype grey_cornertype;
	MPI_Datatype rgb_rowtype;
	MPI_Datatype rgb_coltype;
	MPI_Datatype rgb_cornertype;
	MPI_Datatype imgRowType;
	MPI_Datatype imgColType;
	MPI_Datatype imgCornerType;

	/* MPI requests and flags for use in MPI_Test */

	/* Send requests*/
	MPI_Request north_send_req;
	MPI_Request south_send_req;
	MPI_Request west_send_req;
	MPI_Request east_send_req;
	MPI_Request neast_send_req;
	MPI_Request seast_send_req;
	MPI_Request nwest_send_req;
	MPI_Request swest_send_req;
	/* Recieve requests */
	MPI_Request north_rcv_req;
	MPI_Request south_rcv_req;
	MPI_Request west_rcv_req;
	MPI_Request east_rcv_req;
	MPI_Request neast_rcv_req;
	MPI_Request seast_rcv_req;
	MPI_Request nwest_rcv_req;
	MPI_Request swest_rcv_req;
	/* Send flags */
	int north_send_flag, south_send_flag, west_send_flag, east_send_flag;
	int neast_send_flag, seast_send_flag, nwest_send_flag, swest_send_flag;
	/* Recieve flags */
	int north_rcv_flag=0, south_rcv_flag=0, west_rcv_flag=0, east_rcv_flag=0;
	int neast_rcv_flag=0, seast_rcv_flag=0, nwest_rcv_flag=0, swest_rcv_flag=0;

	double startTime, endTime, elapsedTime, relapsedTime, welapsedTime, maxTime;

	/* Master process initializes : a) reading parameters from input and b) columns & rows splits */
	if ( procId == 0) {

		// Get-check command line parameters
		get_parameters(argc, argv, &imgName, &totalRows, &totalCols, &maxIters, &convIters, &imgType );

		// get image name number of characters (for broadcasting purposes)
		imgNmLen = strlen(imgName);

		/* Check image file existence and read permissions */
		if ( ! ( !access(imgName, F_OK) && !access(imgName, R_OK) ) ){
			printf("File %s does not exist or you do not have read permissions\n", imgName);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			return EXIT_FAILURE;
		}

		// Calculate number of columns splits
		colSplits = columns_splits(totalRows,totalCols,nProcs);
		//printf("number of column splits calculated = %d\n",colSplits);

		// Check if rows/cols can be divided with number of processes
		if (colSplits<0) {
			fprintf(stderr,"Image array cannot be divided to processes\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			return EXIT_FAILURE;
		}
		// Calculate row splits
		rowSplits = nProcs/colSplits;

	}

	/* Sending parameters to processes */
	MPI_Bcast(&imgNmLen, 1, MPI_INT, 0, MPI_COMM_WORLD); 				// Image Name char length (to allocate space in next broadcast)
	if (procId !=0) { imgName = malloc( (imgNmLen+1) * sizeof(char)); } // Allocate memory for image name
	MPI_Bcast(imgName, imgNmLen+1, MPI_CHAR, 0, MPI_COMM_WORLD); 		// Image Name
	MPI_Bcast(&totalRows, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Total Rows
	MPI_Bcast(&totalCols, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Total Columns
	MPI_Bcast(&maxIters, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Max iterations
	MPI_Bcast(&convIters, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Convergence check iterations
	MPI_Bcast(&imgType, 1, MPI_INT, 0, MPI_COMM_WORLD);					// Image type
	MPI_Bcast(&rowSplits, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Number of Row splits
	MPI_Bcast(&colSplits, 1, MPI_INT, 0, MPI_COMM_WORLD);				// Number of Column splits

	/***Check point 1*** : Output the parameters got or calculated */
	if (procId ==0) {
		printf("trows-tcols-maxiters-conviters-imgtype-rowsplits-colsplits = %d/%d/%d/%d/%d/%d/%d\n",
				totalRows,totalCols,maxIters,convIters,imgType,rowSplits,colSplits);
	}

	/* Calculate local row/column numbers on each process */
	locRows = totalRows/rowSplits ;
	locCols = totalCols/colSplits ;

	/* Create vector and corner types for rows, columns, edges pixels */
	/* MPI vector : (count, blocklength, stride, oldtype, &newtype) */
	int halo=1;
	int imgWidth=0;

	/* Create vector types for sending framing rows, columns, pixels according to image type*/
	if (imgType == IMG_GREY) { halo = 1; } else if (imgType == IMG_RGB ) {halo = 3;}
	imgWidth = (locCols+2)*halo;
	MPI_Type_vector(locRows, halo, (locCols+2)*halo, MPI_BYTE, &imgColType);
	MPI_Type_vector(locCols, halo, halo, MPI_BYTE, &imgRowType);
	MPI_Type_vector(1, halo, halo, MPI_BYTE, &imgCornerType);
	MPI_Type_commit(&imgColType);
	MPI_Type_commit(&imgRowType);
	MPI_Type_commit(&imgCornerType);

	/***Check point 2*** : Allocating image arrays */
	//printf("Process %d : allocating image arrays\n",procId);

	/* Assign pointers to current, temporary and next (calculated in each step) image arrays */
	uint8_t* imgcurr = NULL;		// Current image
	uint8_t* imgnext = NULL;		// Calculated image
	uint8_t* imgtemp = NULL;		// Pointer for switching images after each iteration

	/* Find the initial (in total image) row and column of the current process*/
	int initRow;
	int initCol;
	initRow = (procId / colSplits) * locRows;
	initCol = (procId % colSplits) * locCols;
	//printf("process : %d initRow/initCol/locRows/locCols %d/%d/%d/%d\n", procId, initRow,initCol,locRows,locCols);

	/* Set the neighboring processes number for each process */
	establish_neighbors(procId, totalRows, totalCols, locRows, locCols, initRow, initCol, colSplits,
			&North, &South, &West, &East, &NorthWest, &SouthWest, &NorthEast, &SouthEast);

	/* Allocate memory for image arrays - Checking memory allocation succeeded */
	if ( ! allocate_img_arrays(locRows, locCols, imgType, &imgcurr, &imgnext) ) {
		fprintf(stderr,"Memory not available to be allocated to image arrays\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		return EXIT_FAILURE;
	}

	/* Reference ghost area pixels in global image frame - stationary/not included in send-receive operations
	 * - to the corresponding edges pixels (and sub-pixels) */
/*
	frame_pixels_referencing(North, South, West, East, NorthWest, NorthEast, SouthWest, SouthEast,
			imgType, imgcurr, locRows, locCols);
	frame_pixels_referencing(North, South, West, East, NorthWest, NorthEast, SouthWest, SouthEast,
			imgType, imgnext, locRows, locCols);
*/


	/* Initialize a filter to test */
	float filter [3][3]={{1/16.0, 2/16.0, 1/16.0}, {2/16.0, 4/16.0, 2/16.0}, {1/16.0, 2/16.0, 1/16.0}};
	//filter = malloc (3*sizeof(float*));

	/***Check point*** : reading image */
	//printf("Process %d : reading image... \n",procId);

	/* Assign types for parallel IO (using subarrays) */
	MPI_Datatype filetype;
	MPI_Datatype memtype;

	/* True parallel Reading - start counting time*/
	startTime = MPI_Wtime();
	/* Read image */
	read_image_parallel(MPI_COMM_WORLD, procId, imgName, imgType, imgcurr,
			totalRows, totalCols, rowSplits, colSplits, &filetype, &memtype);
	/* End counting parallel reading time */
	endTime = MPI_Wtime();
	/* Calculate elapsed reading time for each process */
	relapsedTime = endTime - startTime;

	/* Get from all processes the maximum time it took them to finish reading */
	MPI_Reduce(&relapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Print worst time : time needed for parallel reading */
	if (procId == 0) { printf("Time elapsed for parallel reading: %3.2f seconds\n", relapsedTime);}

	/***Check point*** : reading image */
	//printf("Process %d : finished reading image... \n",procId);

	/* Positions holding for each neighbor the initial position in local array to be send-received */
	int npos_S, npos_R, spos_S, spos_R, wpos_S, wpos_R, epos_S, epos_R, nwpos_S, nwpos_R, swpos_S, swpos_R,
	nepos_S, nepos_R, sepos_S, sepos_R;

	/* Assign values to the above neighboring positions */
	npos_S = arraypos(locRows, 1, locRows, locCols, halo);
	npos_R = arraypos(locRows+1, 1, locRows, locCols, halo);
	spos_S = arraypos(1, 1, locRows, locCols, halo);
	spos_R = arraypos(0, 1, locRows, locCols, halo);
	wpos_S = arraypos(1, 1, locRows, locCols, halo);
	wpos_R = arraypos(1, 0, locRows, locCols, halo);
	epos_S = arraypos(1, locCols, locRows, locCols, halo);
	epos_R = arraypos(1, locCols+1, locRows, locCols, halo);
	nwpos_S = arraypos(locRows, 1, locRows, locCols, halo);
	nwpos_R = arraypos(locRows+1, 0, locRows, locCols, halo);
	nepos_S = arraypos(locRows, locCols, locRows, locCols, halo);
	nepos_R = arraypos(locRows+1, locCols, locRows, locCols, halo);
	swpos_S = arraypos(1, 1, locRows, locCols, halo);
	swpos_R = arraypos(0, 0, locRows, locCols, halo);
	sepos_S = arraypos(1, locCols, locRows, locCols, halo);
	sepos_R = arraypos(0, locCols+1, locRows, locCols, halo);

	/* Variables holding whether the calculated new image has changed in some pixel */
	int imgHasChanged=0, changed =0, fullImgChanged=0;

	/* In order to compute once and unify all the edges pixels that need to be copied on each process
	 * with the model (Left part) = (Right part) i.e. :
	 * 'counter' * 'counter_multiplier' + 'standard left part' = 'counter' * 'counter_multiplier' + 'standard right part'
	 * where 'counter' starts from 'cntrmin' and ends at 'cntrmax',
	 * pre-assign values to all variables for each N,S,W,E,NW,NE,SW,SE edges
	 */
/*	int n_edge_cntrmult=-1, n_edge_cntrmin=-1, n_edge_cntrmax=-1, n_edge_stdpart_l=-1, n_edge_stdpart_r=-1;
	int s_edge_cntrmult, s_edge_cntrmin, s_edge_cntrmax, s_edge_stdpart_l, s_edge_stdpart_r=-1;
	int w_edge_cntrmult, w_edge_cntrmin, w_edge_cntrmax, w_edge_stdpart_l, w_edge_stdpart_r;
	int e_edge_cntrmult, e_edge_cntrmin, e_edge_cntrmax, e_edge_stdpart_l, e_edge_stdpart_r;
	int nw_edge_cntrmult, nw_edge_cntrmin, nw_edge_cntrmax, nw_edge_stdpart_l, nw_edge_stdpart_r;
	int ne_edge_cntrmult, ne_edge_cntrmin, ne_edge_cntrmax, ne_edge_stdpart_l, ne_edge_stdpart_r;
	int sw_edge_cntrmult, sw_edge_cntrmin, sw_edge_cntrmax, sw_edge_stdpart_l, sw_edge_stdpart_r;
	int se_edge_cntrmult, se_edge_cntrmin, se_edge_cntrmax, se_edge_stdpart_l, se_edge_stdpart_r;

	 Initialization for getting initial addresses for edges/corners copying
	if (North == -1) getting_addreses_parts_for_copying_edges(NN, imgType, locRows, locCols,
				&n_edge_cntrmin, &n_edge_cntrmax, &n_edge_cntrmult, &n_edge_stdpart_l, &n_edge_stdpart_r);
	if (South == -1) getting_addreses_parts_for_copying_edges(NS, imgType, locRows, locCols,
				&s_edge_cntrmin, &s_edge_cntrmax, &s_edge_cntrmult, &s_edge_stdpart_l, &s_edge_stdpart_r);
	if (West == -1) getting_addreses_parts_for_copying_edges(NW, imgType, locRows, locCols,
				&w_edge_cntrmin, &w_edge_cntrmax, &w_edge_cntrmult, &w_edge_stdpart_l, &w_edge_stdpart_r);
	if (East == -1) getting_addreses_parts_for_copying_edges(NE, imgType, locRows, locCols,
				&e_edge_cntrmin, &e_edge_cntrmax, &e_edge_cntrmult, &e_edge_stdpart_l, &e_edge_stdpart_r);
	if (NorthWest == -1) getting_addreses_parts_for_copying_edges(NNW, imgType, locRows, locCols,
				&nw_edge_cntrmin, &nw_edge_cntrmax, &nw_edge_cntrmult, &nw_edge_stdpart_l, &nw_edge_stdpart_r);
	if (NorthEast == -1) getting_addreses_parts_for_copying_edges(NNE, imgType, locRows, locCols,
				&ne_edge_cntrmin, &ne_edge_cntrmax, &ne_edge_cntrmult, &ne_edge_stdpart_l, &ne_edge_stdpart_r);
	if (SouthWest == -1) getting_addreses_parts_for_copying_edges(NSW, imgType, locRows, locCols,
				&sw_edge_cntrmin, &sw_edge_cntrmax, &sw_edge_cntrmult, &sw_edge_stdpart_l, &sw_edge_stdpart_r);
	if (SouthEast == -1) getting_addreses_parts_for_copying_edges(NSE, imgType, locRows, locCols,
				&se_edge_cntrmin, &se_edge_cntrmax, &se_edge_cntrmult, &se_edge_stdpart_l, &se_edge_stdpart_r);*/

	/* Wait for all processes to come to this point */
	MPI_Barrier(MPI_COMM_WORLD);

	/* Intialize counting iterations counter */
	int iter=0;

	/* Start counting time */
	startTime = MPI_Wtime();

	/* Begin calculating iterations */
	for (iter=0; iter < maxIters; iter++){

		/* Send and receive edges and corners data */
		if (North != -1 ) MPI_Isend(&imgcurr[npos_S], 1, imgRowType, North, 0, MPI_COMM_WORLD, &north_send_req);
		if (North != -1 ) MPI_Irecv(&imgcurr[npos_R], 1, imgRowType, North, 0, MPI_COMM_WORLD, &north_rcv_req);
		if (South != -1 ) MPI_Isend(&imgcurr[spos_S], 1, imgRowType, South, 0, MPI_COMM_WORLD, &south_send_req);
		if (South != -1 ) MPI_Irecv(&imgcurr[spos_R], 1, imgRowType, South, 0, MPI_COMM_WORLD, &south_rcv_req);
		if (West != -1 ) MPI_Isend(&imgcurr[wpos_S], 1, imgColType, West, 0, MPI_COMM_WORLD, &west_send_req);
		if (West != -1 ) MPI_Irecv(&imgcurr[wpos_R], 1, imgColType, West, 0, MPI_COMM_WORLD, &west_rcv_req);
		if (East != -1 ) MPI_Isend(&imgcurr[epos_S], 1, imgColType, East, 0, MPI_COMM_WORLD, &east_send_req);
		if (East != -1 ) MPI_Irecv(&imgcurr[epos_R], 1, imgColType, East, 0, MPI_COMM_WORLD, &east_rcv_req);
		if (NorthWest != -1 ) MPI_Isend(&imgcurr[nwpos_S], 1, imgCornerType, NorthWest, 0, MPI_COMM_WORLD, &nwest_send_req);
		if (NorthWest != -1 ) MPI_Irecv(&imgcurr[nwpos_R], 1, imgCornerType, NorthWest, 0, MPI_COMM_WORLD, &nwest_rcv_req);
		if (SouthWest != -1 ) MPI_Isend(&imgcurr[swpos_S], 1, imgCornerType, SouthWest, 0, MPI_COMM_WORLD, &swest_send_req);
		if (SouthWest != -1 ) MPI_Irecv(&imgcurr[swpos_R], 1, imgCornerType, SouthWest, 0, MPI_COMM_WORLD, &swest_rcv_req);
		if (NorthEast != -1 ) MPI_Isend(&imgcurr[nepos_S], 1, imgCornerType, NorthEast, 0, MPI_COMM_WORLD, &neast_send_req);
		if (NorthEast != -1 ) MPI_Irecv(&imgcurr[nepos_R], 1, imgCornerType, NorthEast, 0, MPI_COMM_WORLD, &neast_rcv_req);
		if (SouthEast != -1 ) MPI_Isend(&imgcurr[sepos_S], 1, imgCornerType, SouthEast, 0, MPI_COMM_WORLD, &seast_send_req);
		if (SouthEast != -1 ) MPI_Irecv(&imgcurr[sepos_R], 1, imgCornerType, SouthEast, 0, MPI_COMM_WORLD, &seast_rcv_req);

		 /* Calculate convolution for non-edges pixels (start row/column : for first row/column equals 1
		    last row/column : equals locRows and locCols respectively) */
		//printf("Process %d calc main convolution set begins - iter %d\n", procId, iter);
		changed = calculate_convolution(imgcurr, imgnext, 2, locRows-1, 2, locCols-1,
				locCols, imgType, filter, thread_count);
		//printf("Process %d calc main convolution set ends - iter %d\n", procId, iter);

		/* Set bit if image has changed during convolution */
		if (changed){imgHasChanged = 1;}

		/* If no neighbors are present set their receive flag to TRUE */
		if (North == -1){ north_rcv_flag = 1;}
		if (South == -1){ south_rcv_flag = 1;}
		if (West == -1){ west_rcv_flag = 1;}
		if (East == -1){ east_rcv_flag = 1;}
		if (NorthWest == -1){ nwest_rcv_flag = 1;}
		if (SouthWest == -1){ swest_rcv_flag = 1;}
		if (NorthEast == -1){ neast_rcv_flag = 1;}
		if (SouthEast == -1){ seast_rcv_flag = 1;}

		/* Wait for receiving and calculate convolution of edges and corner pixels */
		while ( ! ( north_rcv_flag && south_rcv_flag && west_rcv_flag && east_rcv_flag &&
				nwest_rcv_flag && neast_rcv_flag && swest_rcv_flag && seast_rcv_flag) ) {

			/****** NORTH SIDE ******/
			/* Check if edge was received */
			if (North != -1 ) {MPI_Test(&north_rcv_req, &north_rcv_flag, MPI_STATUS_IGNORE);}
			/* Calc convolution on edge */
			if ( north_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, locRows, locRows, 2,
						locCols-1, locCols, imgType, filter, thread_count);
				/* Copy edge to frame if no neighbor is present */
				if (North == -1 ){
					for (i=halo;i<imgWidth-halo;i++) {
						imgnext[(locRows+1)*imgWidth + i] = imgnext[(locRows)*imgWidth + i];
					}
				}
				if (changed){imgHasChanged = 1;}

				/* Copying relevant edge pixels to frame if process works at image frame */
				/*
					if (nptr_R == NULL) {
					copy_edges_to_frame(imgnext, imgType, n_edge_cntrmin, n_edge_cntrmax, n_edge_cntrmult,
						n_edge_stdpart_l, n_edge_stdpart_r);
					}
				 */
			}

			/****** SOUTH SIDE ******/
			if (South != -1 ) {	MPI_Test(&south_rcv_req, &south_rcv_flag, MPI_STATUS_IGNORE); }
			/* If yes calc edge convolution */
			if ( south_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, 1, 1, 2, locCols-1, locCols,
						imgType, filter, thread_count);
				if (South == -1 ){
					for (i=halo;i<imgWidth-halo;i++) {
						imgnext[i] = imgnext[imgWidth + i];
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
				 	 Copying relevant edge pixels to
					if (sptr_S == NULL ) {
						copy_edges_to_frame(imgnext, imgType, s_edge_cntrmin, s_edge_cntrmax, s_edge_cntrmult,
							s_edge_stdpart_l, s_edge_stdpart_r);
					}
				 */
			}



			/****** WEST SIDE ******/
			if (West != -1 ) { MPI_Test(&west_rcv_req, &west_rcv_flag, MPI_STATUS_IGNORE); }
			/* If yes calc edge convolution */
			if ( west_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, 2, locRows-1, 1, 1, locCols,
						imgType, filter, thread_count);
				if (West == -1 ){
					for (i=1 ; i<locRows+1 ; i++) {
						for (k=0; k<halo; k++){
							imgnext[i*imgWidth + k] = imgnext[i*imgWidth + halo + k];
						}
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
				 	 Copying relevant edge pixels to frame if process works at image frame
					if (wptr_R == NULL) {
						copy_edges_to_frame(imgnext, imgType, w_edge_cntrmin, w_edge_cntrmax, w_edge_cntrmult,
						w_edge_stdpart_l, w_edge_stdpart_r);
					}
				 */
			}

			/****** EAST SIDE ******/
			if (East != -1 ) { MPI_Test(&east_rcv_req, &east_rcv_flag, MPI_STATUS_IGNORE); }
			/* If yes calc edge convolution */
			if ( east_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, 2, locRows-1, locCols, locCols, locCols,
						imgType, filter, thread_count);
				if (East == -1 ){
					for (i=1 ; i<locRows+1 ; i++) {
						for (k=0; k<halo; k++){
							imgnext[((i+1)*imgWidth - halo) + k] = imgnext[((i+1)*imgWidth - 2*halo) + k];
						}
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
				 	 Copying relevant edge pixels to frame if process works at image frame
					if (eptr_R == NULL) {
					copy_edges_to_frame(imgnext, imgType, e_edge_cntrmin, e_edge_cntrmax, e_edge_cntrmult,
						e_edge_stdpart_l, e_edge_stdpart_r);
					}
				 */
			}

			/****** NORTH-WEST SIDE (corner pixel) ******/
			if (NorthWest != -1 ) {	MPI_Test(&nwest_rcv_req, &nwest_rcv_flag, MPI_STATUS_IGNORE); }
			/* If yes calc edge convolution */
			if ( nwest_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, locRows, locRows, 1, 1, locCols,
						imgType, filter, thread_count);
				if (NorthWest == -1 ){
					for (k=0; k<halo; k++){
						imgnext[(locRows+1)*imgWidth + k] = imgnext[(locRows)*imgWidth + halo + k];
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
					 	 Copying relevant edge pixels to frame if process works at image frame
						if (nwptr_R == NULL) {
							copy_edges_to_frame(imgnext, imgType, nw_edge_cntrmin, nw_edge_cntrmax, nw_edge_cntrmult,
								nw_edge_stdpart_l, nw_edge_stdpart_r);
						}
				 */
			}

			/****** NORTH-EAST SIDE (corner pixel) ******/
			if (NorthEast != -1 ) { MPI_Test(&neast_rcv_req, &neast_rcv_flag, MPI_STATUS_IGNORE);}
			/* If yes calc edge convolution */
			if ( neast_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, locRows, locRows, locCols, locCols, locCols,
						imgType, filter, thread_count);
				if (NorthEast == -1 ){
					for (k=0; k<halo; k++){
						imgnext[(locRows+2)*imgWidth - halo + k] = imgnext[(locRows+1)*imgWidth - 2*halo + k];
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
				 	 Copying relevant edge pixels to frame if process works at image frame
					if (neptr_R == NULL) {
						copy_edges_to_frame(imgnext, imgType, ne_edge_cntrmin, ne_edge_cntrmax, ne_edge_cntrmult,
							ne_edge_stdpart_l, ne_edge_stdpart_r);
					}
				 */
			}


			/****** SOUTH-WEST SIDE (corner pixel) ******/
			if (SouthWest != -1 ) {	MPI_Test(&swest_rcv_req, &swest_rcv_flag, MPI_STATUS_IGNORE); }
			/* If yes calc edge convolution */
			if ( swest_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, 1, 1, 1, 1, locCols,
						imgType, filter, thread_count);
				if (SouthWest == -1 ){
					for (k=0; k<halo; k++){
						imgnext[k] = imgnext[imgWidth + halo + k];
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
				 	Copying relevant edge pixels to frame if process works at image frame
					if (swptr_R == NULL) {
						copy_edges_to_frame(imgnext, imgType, sw_edge_cntrmin, sw_edge_cntrmax, sw_edge_cntrmult,
							sw_edge_stdpart_l, sw_edge_stdpart_r);
					}
				 */
			}

			/****** SOUTH-EAST SIDE (corner pixel) ******/
			if (SouthEast != -1 ) {	MPI_Test(&seast_rcv_req, &seast_rcv_flag, MPI_STATUS_IGNORE); }
			/* If yes calc edge convolution */
			if ( seast_rcv_flag ) {
				changed = calculate_convolution(imgcurr, imgnext, 1, 1, locCols, locCols, locCols,
						imgType, filter, thread_count);
				if (SouthEast == -1 ){
					for (k=0; k<halo; k++){
						imgnext[imgWidth - halo + k] = imgnext[1*imgWidth - 2*halo + k];
					}
				}
				if (changed){imgHasChanged = 1;}
				/*
				 	 Copying relevant edge pixels to frame if process works at image frame
					if (septr_R == NULL) {
						copy_edges_to_frame(imgnext, imgType, se_edge_cntrmin, se_edge_cntrmax, se_edge_cntrmult,
							se_edge_stdpart_l, se_edge_stdpart_r);
					}
				 */
			}

		}

		/* Reset receive edges flags to FALSE */
		north_rcv_flag = south_rcv_flag = west_rcv_flag = east_rcv_flag = nwest_rcv_flag =
				nwest_rcv_flag = swest_rcv_flag = neast_rcv_flag = seast_rcv_flag = 0;

		 /* Ensure sending of edges/corners has been completed in order to proceed*/
		if (North != -1 ) MPI_Wait(&north_send_req, MPI_STATUS_IGNORE);
		if (South != -1 ) MPI_Wait(&south_send_req, MPI_STATUS_IGNORE);
		if (West != -1 ) MPI_Wait(&west_send_req, MPI_STATUS_IGNORE);
		if (East != -1 ) MPI_Wait(&east_send_req, MPI_STATUS_IGNORE);
		if (NorthWest != -1 ) MPI_Wait(&nwest_send_req, MPI_STATUS_IGNORE);
		if (SouthWest != -1 ) MPI_Wait(&swest_send_req, MPI_STATUS_IGNORE);
		if (NorthEast != -1 ) MPI_Wait(&neast_send_req, MPI_STATUS_IGNORE);
		if (SouthEast != -1 ) MPI_Wait(&seast_send_req, MPI_STATUS_IGNORE);

		/* Change between t0 and t1 arrays in each time step*/
		imgtemp = imgcurr;
		imgcurr = imgnext;
		imgnext = imgtemp;

		/* Convergence check */
		if (iter % convIters == 0) {
			/* Get 'image changed' variable from all processes (reduce to total variable) */
			MPI_Allreduce(&imgHasChanged, &fullImgChanged, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			/* Act if not changed : message and (possibly) exit for loop*/
			if ((!fullImgChanged) && procId ==0 ){
				printf(" Image has no change after iteration %d. Terminating convolution steps...\n", iter);
				break;
			}
		}
	}

	/* End counting time */
	endTime = MPI_Wtime();
	/* Calculate elapsed time for each process */
	elapsedTime = endTime - startTime;

	/*** Check point *** : Confirm given iterations have been performed */
	//printf("Final iteration : %d\n", iter);

	/* Get from all processes the maximum time it took them to finish */
	MPI_Reduce(&elapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	/* Print worst time : time needed for parallel computations */
	if (procId == 0) { printf("time elapsed : %3.2f seconds\n", elapsedTime);}

	/* Start counting time for parallel writing*/
	startTime = MPI_Wtime();

	/* Write convoluted image to file - true parallel IO */
	write_image_parallel(MPI_COMM_WORLD, imgName, imgcurr, &filetype, &memtype);

	/* End counting parallel writing time */
	endTime = MPI_Wtime();
	/* Calculate elapsed writing time for each process */
	welapsedTime = endTime - startTime;

	/* Get from all processes the maximum time it took them to finish writing */
	MPI_Reduce(&welapsedTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	/* Print worst time : time needed for parallel writing */
	if (procId == 0) {
		printf("Time elapsed for parallel writing: %3.2f seconds\n", welapsedTime);
		//printf("num threads : %d\n", omp_get_num_threads());
		print_log_to_file(nProcs, imgName, totalRows, totalCols, maxIters, convIters,
				relapsedTime, elapsedTime, welapsedTime, thread_count);
	}

	free(imgcurr);
	free(imgnext);

/*
	#pragma omp parallel
	{
	printf("thread num %d\n", omp_get_thread_num());
	}
*/

	MPI_Finalize();

	return EXIT_SUCCESS;

	return 0;

}  /* main */
/*************************************************************************************************************/
/************************************* MAIN END **************************************************************/
/*************************************************************************************************************/

void print_log_to_file(int nProcs, char* imgName, int nrows, int ncols, int maxiters, int conviters,
		double readtime, double runtime, double writetime, int nthreads ){

	FILE* fl;

	fl = fopen(".omp_mpi_log", "a+");

	if (fl==NULL) exit(-1);

	struct tm* currtime;
	currtime = mycurrtime();

	char logline[300];
	sprintf(logline,"\n%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%3.2f secs\t%3.2f secs\t%3.2f secs",
			strtok(asctime(currtime), "\n"), imgName, nrows, ncols, maxiters, conviters,
			nProcs, nthreads, runtime, readtime, writetime);
	fputs(logline, fl);

	fclose(fl);

}

struct tm* mycurrtime()
{
  time_t rawtime;
  struct tm * timeinfo;

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  //printf ( "Current local time and date: %s", asctime (timeinfo) );

  return timeinfo;
}

int arraypos(int row, int col , int locRows, int locCols, int halo){

	int pos, width;

	/* Calc enhanced image width (with ghost area) in columns/subpixels */
	width = (locCols+2)*halo;
	/* Calc position */
	pos = row * width + col * halo;

	return pos;
}

/* Reference pixels at ghost area where there are no neighbors to the pixels next to them (to avoid copying) */
/*
void frame_pixels_referencing(int North, int South, int West, int East, int NorthWest, int NorthEast,
		int SouthWest, int SouthEast, int imgType, uint8_t* locImg, int locRows, int locCols){

	int halo;
	int i;
	int enhRows, enhCols, initCol, endCol, initRow, endRow, width;
	int row,col, start_point, dest_point, rowpoint;

	 Set the halo/ghost are in each left-right side of the local image
	if (imgType == IMG_GREY ) {halo=1;} else if (imgType == IMG_RGB) {halo=3;}

	 Number of local rows including ghost area
	enhRows = locRows+2;
	 Number of local columns including ghost area
	enhCols = (locCols+2)*halo;
	 Width of local image array in 'sub'-pixels (columns)
	width = enhCols;
	 Set the column number on the left that the image starts excluding ghost area
	initCol = halo;
	 Set the column number that the ghost area on the right starts ( -1 the last image column)
	endCol = enhCols-halo;
	 Set the row number on the lower side that the image starts excluding ghost area
	initRow = 1;
	 Set the row number on the upper side that the ghost area starts ( -1 the last image row)
	endRow = enhRows - 1;

	if (North==-1){
		 Set the starting cell of the frame and the destination starting point
		start_point = endRow*width;
		dest_point = (endRow-1)*width;
		 Assign the pointer of each frame pixel to the destination pixel
		for (col = initCol ; col < endCol; col++ ){
			&(locImg[start_point + col]) = &(locImg[dest_point + col]);
		}
	}

	if (South == -1 ){
		 Set the starting cell of the frame and the destination starting point
		start_point = 0*width;
		dest_point = 1*width;
		 Assign the pointer of each frame pixel to the destination pixel
		for (col = initCol ; col < endCol; col++ ){
			&locImg[start_point + col] = &locImg[dest_point + col];
		}
	}

	if (West == -1 ){
		 Set the starting cell of the frame and the destination starting point
		start_point = 0*width;
		dest_point = 0*width + halo;
		 Assign the pointer of each frame pixel to the destination pixel
		for (row = initRow ; row < endRow; row++ ){
			 Climbing in west column starting point in rows moves
			rowpoint = row*width;
			 Assign the pointers to each subpixel in halo
			for (i=0;i<halo;i++){
					&locImg[start_point + rowpoint + i] = &locImg[dest_point + rowpoint + i];
			}
		}
	}

	if (East == -1 ){
		 Set the starting cell of the frame and the destination starting point
		start_point = endCol;
		dest_point = endCol - halo;
		 Assign the pointer of each frame pixel to the destination pixel
		for (row = initRow ; row < endRow; row++ ){
			 Climbing in west column starting point in rows moves
			rowpoint = row*width;
			 Assign the pointers to each subpixel in halo
			for (i=0;i<halo;i++){
					&locImg[start_point + rowpoint + i] = &locImg[dest_point + rowpoint + i];
			}
		}
	}

	if (NorthWest==-1){
		 Set the starting cell of the frame and the destination starting point
		start_point = endRow*width;
		dest_point = (endRow-1)*width;
		 Assign the pointer of each frame pixel to the destination pixel
		for (i=0 ; i<halo; i++ ){
			&locImg[start_point + i] = &locImg[dest_point + i];
		}
	}

	if (SouthWest == -1 ){
		 Set the starting cell of the frame and the destination starting point
		start_point = 0*width;
		dest_point = 1*width;
		 Assign the pointer of each frame pixel to the destination pixel
		for (i=0 ; i<halo; i++ ){
			&locImg[start_point + i] = &locImg[dest_point + i];
		}
	}

	if (NorthEast==-1){
		 Set the starting cell of the frame and the destination starting point
		start_point = endRow*width + endCol;
		dest_point = (endRow-1)*width + endCol - halo;
		 Assign the pointer of each frame pixel to the destination pixel
		for (i=0 ; i<halo; i++ ){
			&locImg[start_point + i] = &locImg[dest_point + i];
		}
	}

	if (SouthEast == -1 ){
		 Set the starting cell of the frame and the destination starting point
		start_point = 0*width + endCol;
		dest_point = 1*width + endCol - halo;
		 Assign the pointer of each frame pixel to the destination pixel
		for (i=0 ; i<halo; i++ ){
			&locImg[start_point + i] = &locImg[dest_point + i];
		}
	}

}
*/

/* True parallel Image Reading (parallel IO) */
void write_image_parallel(MPI_Comm comm, char* imgName, uint8_t* img, MPI_Datatype* filetype, MPI_Datatype* memtype){
	MPI_Status status;
	MPI_File fh;
	MPI_Offset offset;
	int count;
	int procId;

	/* Set the new convoluted image file name */
	size_t len = strlen("conv_") + strlen(imgName) + 1 ;
	char* filtImgName = malloc(len);
	sprintf(filtImgName, "conv_%s", imgName);

	/* Open file for multiple IO */
	MPI_File_open(comm, filtImgName, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	/* Set it's view as the above set 2dim local array
	 * (so file is shown as a local array [locRows]*[locCols] */
	MPI_File_set_view(fh, 0, MPI_BYTE, *filetype, "native", MPI_INFO_NULL);

	/* Read in 'augmented' local array in memory the local array part according to the file view */
	MPI_File_write_all(fh, img, 1, *memtype, &status);
	MPI_Get_count(&status, *memtype, &count);
	MPI_Comm_rank(comm, &procId);
	//printf("process %d - write %d memtype \n", procId, count);
	/* Close the file */
	MPI_File_close(&fh);


}

/* True parallel Image Reading (parallel IO) */
void read_image_parallel(MPI_Comm comm, int procId, char* imgName, int imgType, uint8_t* img,
		int totalRows, int totalCols, int rowSplits, int colSplits , MPI_Datatype* filetype, MPI_Datatype* memtype)
		{

	int gsizes[2];
	int psizes[2];
	int lsizes[2];
	int memsizes[2];
	int start_indices[2];
	int coords[2];
	MPI_Status status;
	MPI_File fh;
	MPI_Offset offset;
	int count;
	int halo;
	char* cpoint;
	char* dpoint;
	//MPI_Datatype filetype;
	//MPI_Datatype memtype;

	/* Number of items per pixel and also 'halo'/'ghost' area size */
	if (imgType==IMG_GREY) { halo = 1; } else if (imgType == IMG_RGB ) { halo = 3; }
	/* Set global array sizes in each dimension (row/column) */
	gsizes[0]=totalRows;
	gsizes[1]=halo*totalCols;
	/* Set number of rows and columns splits*/
	psizes[0]=rowSplits;
	psizes[1]=colSplits;
	/* Calculate local array sizes in each dimension (row/column)*/
	lsizes[0]=gsizes[0]/psizes[0];
	lsizes[1]=gsizes[1]/psizes[1];
	int locRows,locCols;
	locRows = lsizes[0];
	locCols = lsizes[1];
	/* Calculate current process coordinates in each dimension (x,y) */
	coords[0] = (procId / colSplits);   // Row process  (e.g (1,2) process at grid position coords(1,2))
	coords[1] = (procId % colSplits);	// Column process
	/* Calculate where the first element of the local array is within the global array */
	start_indices[0] = coords[0]*lsizes[0];
	start_indices[1] = coords[1]*lsizes[1];

	/* Create a 2dim subarray that splits the global size (2dim like) array in (2dim like) smaller
	 * local arrays per process - assign a new type to it */
	MPI_Type_create_subarray(2, gsizes, lsizes, start_indices, MPI_ORDER_C, MPI_BYTE, filetype);
	MPI_Type_commit(filetype);

	/* Open file for multiple IO */
	MPI_File_open(comm, imgName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	/* Set it's view as the above set 2dim local array
	 * (so file is shown as a local array [locRows]*[locCols] */
	MPI_File_set_view(fh, 0, MPI_BYTE, *filetype, "native", MPI_INFO_NULL);

	/* Set memory sizes of the type you need - local array with 'halo'/'ghost' area according to image type*/
	memsizes[0]=lsizes[0] + 2;
	memsizes[1]=lsizes[1] + 2 * halo;
	/* Set the starting point of the 'augmented' local array accordingly in each dimension */
	start_indices[0]= 1;
	start_indices[1]= halo;
	/* Create an 'augmented' (with halo) version of the local array and assign a new type to it */
	MPI_Type_create_subarray(2, memsizes, lsizes, start_indices, MPI_ORDER_C, MPI_BYTE, memtype);
	MPI_Type_commit(memtype);

	/* Set the point in img to read from */
	//cpoint = img;

	/* Read in 'augmented' local array in memory the local array part according to the file view */
	MPI_File_read_all(fh, img, 1, *memtype, &status);
	MPI_Get_count(&status, *memtype, &count);
	//printf("process %d - read %d memtype \n", procId, count);

	/* Close the file */
	MPI_File_close(&fh);

/*

	// Set the edges to the neighbors values
	// Left and Right column (not including row 0 and row locRows+1)
	dpoint = &img[locCols+2] ;
	cpoint = &img[locCols+2 +1] ;
	MPI_Sendrecv(cpoint,1, grey_coltype, pid, 0,
			dpoint, 1, grey_coltype, pid, 0, comm,MPI_STATUS_IGNORE);

	dpoint = &img[2*(locCols+2) -1];
	cpoint = &img[2*(locCols+2) -2];
	MPI_Sendrecv(cpoint,1, grey_coltype, pid, 0,
			dpoint, 1, grey_coltype, pid, 0, comm,MPI_STATUS_IGNORE);
	//printf(" %d = %d\n", img[2*(locCols+2)],img[2*(locCols+2)+1]);

	// Top and Down row (not including column 0 and column locCols+1)
	dpoint = &img[1];
	cpoint = &img[locCols+2 +1];
	MPI_Sendrecv(cpoint,1, grey_rowtype, pid, 0,
			dpoint, 1, grey_rowtype, pid, 0, comm,MPI_STATUS_IGNORE);

	dpoint = &img[locRows * (locCols+2) +1 ];
	cpoint = &img[(locRows-1)*(locCols+2) +1 ];
	MPI_Sendrecv(cpoint,1, grey_rowtype, pid, 0,
			dpoint, 1, grey_rowtype, pid, 0, comm,MPI_STATUS_IGNORE);

	// Corner pixels
	img[0] = img[1]; 													// left-down pixel
	img[locCols+2 -1] = img[locCols+2 -2];								// right-down pixel
	img[locRows*(locCols+2)] = img[locRows*(locCols+2) +1];				// left-up pixel
	img[(locRows+1)*(locCols+2) -1] = img[(locRows+1)*(locCols+2) -2];	// right-up pixel
*/


}

void getting_addreses_parts_for_copying_edges(neighbor_orientation edge, int imgType, int locRows, int locCols,
		int* cntr_min, int* cntr_max, int* cntr_mult, int* stdpart_l, int* stdpart_r){

	int mult;

	/* Set the multiplication factor to find proper row place */
	if (imgType == IMG_GREY){
		mult = locCols + 2;
	} else if (imgType == IMG_RGB) {
		mult = 3 * (locCols+2);
	}

	/* Set counter min, max, multiplier and steady parts of right and left addresses according
	 * to neighbor orientation */
	switch(edge) {
		case NN:
			*cntr_min = 1 ;
			*cntr_max = locCols;
			*cntr_mult = 1;
			*stdpart_l = locRows*mult;
			*stdpart_r = (locRows-1)*mult;
			break;
		case NS:
			*cntr_min = 1 ;
			*cntr_max = locCols;
			*cntr_mult = 1;
			*stdpart_l = 0;
			*stdpart_r = 1*mult;
			break;
		case NW :
			*cntr_min = 1 ;
			*cntr_max = locRows-1;
			*cntr_mult = mult;
			*stdpart_l = 0;
			*stdpart_r = 1;
			break;
		case NE :
			*cntr_min = 1 ;
			*cntr_max = locRows-1;
			*cntr_mult = mult;
			*stdpart_l = locCols+1;
			*stdpart_r = locCols;
			break;
		case NNW :
			*cntr_min = 0 ;
			*cntr_max = 0;
			*cntr_mult = 0;
			*stdpart_l = locRows*mult;
			*stdpart_r = (locRows-1)*mult;
			break;
		case NSW :
			*cntr_min = 0 ;
			*cntr_max = 0;
			*cntr_mult = 0;
			*stdpart_l = 0*mult;
			*stdpart_r = 1*mult + 1;
			break;
		case NNE :
			*cntr_min = 0 ;
			*cntr_max = 0;
			*cntr_mult = 0;
			*stdpart_l = locRows*mult + (locCols+1);
			*stdpart_r = (locRows-1)*mult + locCols;
			break;
		case NSE :
			*cntr_min = 0 ;
			*cntr_max = 0;
			*cntr_mult = 0;
			*stdpart_l = 0*mult + (locCols+1);
			*stdpart_r = 1*mult + locCols;
			break;
	}
}

void copy_edges_to_frame(char* imgnext, int imgType, int cntr_min, int cntr_max,
		int cntr_mult, int stdpart_l, int stdpart_r){

	int cntr;

	if (imgType == IMG_GREY){

		for (cntr = cntr_min; cntr <= cntr_max ; cntr++ ){
			imgnext[stdpart_l + cntr * cntr_mult] = imgnext[ stdpart_r + cntr * cntr_mult];
		}

	} else if (imgType == IMG_RGB) {

		int j;

		for (cntr = cntr_min; cntr <= cntr_max ; cntr++ ){

			for(j=0;j<=2;j++){
				imgnext[stdpart_l + cntr * cntr_mult + j] = imgnext[ stdpart_r + cntr * cntr_mult + j];
			}
		}

	}
}

inline int calculate_convolution(uint8_t* imgcurr, uint8_t* imgnext, int startRow, int endRow, int startCol,
		int endCol, int locCols, int imgType, float filter[3][3], int thread_count){

	int row, col, tot, startTot, endTot;
	int i,j,k;
	int s=1;
	int imgChanged = 0;
	int imgWidth=0, halo=1;

	/* Set the appropriate halo*/
	if(imgType == IMG_GREY){ halo = 1;} else if (imgType == IMG_RGB ){ halo =3;}

	/* Set variables to hold subpixels previous and next values */
	float prevres[halo];
	float nextres[halo];

	/* Find the image width in columns/subpixels according to image type */
	imgWidth = (locCols+2) * halo;

# pragma omp parallel for num_threads( thread_count ) schedule(static,1) \
		default(none) reduction(max: imgChanged) \
		private(row, col, k, i, j, prevres, nextres) \
		shared(s, imgWidth,startRow, endRow, startCol, endCol, halo, imgcurr, imgnext, filter) \
		collapse(2)
	for ( row = startRow; row <= endRow ; row++){
		for ( col = startCol; col <= endCol ; col++ ) {
			// Get previous pixel value
			for (k=0;k<halo;k++){
				prevres[k] = imgcurr[imgWidth*row + halo*col + k];
				nextres[k] = 0.0;
			}
			// Calculate current pixel convolution
			for (i=-s; i <= +s; i++){
				for (j=-s; j <= +s; j++){
					// Add each contribution to result
					for (k=0;k<halo;k++){
						nextres[k] += imgcurr[imgWidth*(row-i) + halo*(col-j) + k] * filter[1+i][1+j];
					}
				}
			}
			// Check if pixel has changed
			for (k=0;k<halo;k++){
				if ( nextres[k] != prevres[k]) imgChanged = 1;
			}
			// Set pixel new value to next image array
			for (k=0;k<halo;k++){
				imgnext[imgWidth*row + halo*col + k] = nextres[k];
			}
		}
	}
	/* Return 1 if image has changed */
	return imgChanged;
}

void initialize_neighbors_pointers_in_image_array(char* imgcurr, int locRows, int locCols,int imgType,
		int North, int South, int West, int East, int NorthWest, int NorthEast, int SouthWest, int SouthEast,
		char** nptr_S, char** nptr_R, char** sptr_S, char** sptr_R, char** wptr_S, char** wptr_R,
		char** eptr_S, char** eptr_R, char** swptr_S, char** swptr_R, char** septr_S, char** septr_R,
		char** nwptr_S, char** nwptr_R, char** neptr_S, char** neptr_R){

	if (imgType == IMG_GREY) {

		if ( North !=-1 ) {
			*nptr_S = positionptr(imgcurr, locRows, 1, locCols+2);
			*nptr_R = positionptr(imgcurr, locRows+1, 1, locCols+2);
		}
		if ( South !=-1 ){
			*sptr_S = positionptr(imgcurr, 1, 1, locCols+2);
			*sptr_R = positionptr(imgcurr, 0, 1, locCols+2);
		}
		if ( West !=-1 ){
			*wptr_S = positionptr(imgcurr, 1, 1, locCols+2);
			*wptr_R = positionptr(imgcurr, 1, 0, locCols+2);
		}
		if ( East !=-1 ){
			*eptr_S = positionptr(imgcurr, 1, locCols, locCols+2);
			*eptr_R = positionptr(imgcurr, 1, locCols+1, locCols+2);
		}
		if ( NorthWest !=-1 ) {
			*nwptr_S = positionptr(imgcurr, locRows, 1, locCols+2);
			*nwptr_R = positionptr(imgcurr, locRows+1, 0, locCols+2);
		}
		if ( NorthEast !=-1 ) {
			*neptr_S = positionptr(imgcurr, locRows, locCols, locCols+2);
			*neptr_R = positionptr(imgcurr, locRows+1, locCols+1, locCols+2);
		}
		if ( SouthWest !=-1 ){
			*swptr_S = positionptr(imgcurr, 1, 1, locCols+2);
			*swptr_R = positionptr(imgcurr, 0, 0, locCols+2);
		}
		if ( SouthEast !=-1 ){
			*septr_S = positionptr(imgcurr, 1, locCols, locCols+2);
			*septr_R = positionptr(imgcurr, 0, locCols+1, locCols+2);
		}

	} else if (imgType == IMG_RGB) {

		if ( North !=-1 ) {
			*nptr_S = positionptr(imgcurr, locRows, 1*3, 3*(locCols+2));
			*nptr_R = positionptr(imgcurr, locRows+1, 1*3, 3*(locCols+2));
		}
		if ( South !=-1 ){
			*sptr_S = positionptr(imgcurr, 1, 1*3, 3*(locCols+2));
			*sptr_R = positionptr(imgcurr, 0, 1*3, 3*(locCols+2));
		}
		if ( West !=-1 ){
			*wptr_S = positionptr(imgcurr, 1, 1*3, 3*(locCols+2));
			*wptr_R = positionptr(imgcurr, 1, 0*3, 3*(locCols+2));
		}
		if ( East !=-1 ){
			*eptr_S = positionptr(imgcurr, 1, locCols*3, 3*(locCols+2));
			*eptr_R = positionptr(imgcurr, 1, (locCols+1)*3, 3*(locCols+2));
		}
		if ( NorthWest !=-1 ) {
			*nwptr_S = positionptr(imgcurr, locRows, 1*3, 3*(locCols+2));
			*nwptr_R = positionptr(imgcurr, locRows+1, 0*3, 3*(locCols+2));
		}
		if ( NorthEast !=-1 ) {
			*neptr_S = positionptr(imgcurr, locRows, 3*locCols, 3*(locCols+2));
			*neptr_R = positionptr(imgcurr, locRows+1, 3*(locCols+1), 3*(locCols+2));
		}
		if ( SouthWest !=-1 ){
			*swptr_S = positionptr(imgcurr, 1, 3*1, 3*(locCols+2));
			*swptr_R = positionptr(imgcurr, 0, 3*0, 3*(locCols+2));
		}
		if ( SouthEast !=-1 ){
			*septr_S = positionptr(imgcurr, 1, 3*locCols, 3*(locCols+2));
			*septr_R = positionptr(imgcurr, 0, 3*(locCols+1), 3*(locCols+2));
		}
	}

}

char* positionptr(char* img, int cRow, int cCol, int cols){
	// return the position in image array for row/column cRow/cCol
	return &img[cRow * cols + cCol];
}

void establish_neighbors(int pid, int totalRows, int totalCols,
		int locRows, int locCols, int initRow, int initCol, int colSplits,
		int* north, int* south, int* west, int* east,
		int* nwest, int* swest, int* neast, int* seast){

	if (initRow + locRows < totalRows){
		*north = pid + colSplits;
	}
	if (initRow != 0){
		*south = pid - colSplits;
	}
	if (initCol != 0) {
		*west = pid - 1;
	}
	if (initCol + locCols < totalCols ) {
		*east = pid + 1;
	}

	if (*north>0 && *west>0){
		//printf("pr N : %d - W : %d\n", *north, *west);
		*nwest = *north-1;
	}
	if (*north>0 && *east>0){
		*neast = *north+1;
	}

	if (*south>0 && *west>0){
		*swest = *south-1;
	}
	if (*south>0 && *east>0){
		*seast = *south+1;
	}

/*
	printf("process %d - North =%d\n",pid,*north);
	printf("process %d - South =%d\n",pid,*south);
	printf("process %d - West =%d\n",pid,*west);
	printf("process %d - East =%d\n",pid,*east);
	printf("process %d - NorthWest =%d\n",pid,*nwest);
	printf("process %d - NorthEast =%d\n",pid,*neast);
	printf("process %d - SouthWest =%d\n",pid,*swest);
	printf("process %d - SouthEast =%d\n",pid,*seast);
*/
}

void read_full_image(MPI_Comm comm, int pid, char* imgName, int imgType, char* img,
		int initRow, int initCol, int locRows, int locCols, int totalRows, int totalCols,
		MPI_Datatype grey_rowtype, MPI_Datatype grey_coltype, MPI_Datatype rgb_rowtype,
		MPI_Datatype rgb_coltype){

	int cRow;
	int i,j;
	char* cpoint;
	char* dpoint;
	MPI_File imgf;

	// Open file for parallel reading
	MPI_File_open(comm, imgName, MPI_MODE_RDONLY, MPI_INFO_NULL, &imgf);

	if (imgType == IMG_GREY){

		for ( cRow = 0; cRow<locRows; cRow++){
			MPI_File_seek(imgf, (initRow + cRow) * totalCols + initCol, MPI_SEEK_SET);
			cpoint = &img[ cRow * (locCols+2) + 1 ];
			MPI_File_read(imgf, cpoint, locCols, MPI_BYTE, MPI_STATUS_IGNORE);
		}

		// Set the edges to the neighbors values
		// Left and Right column (not including row 0 and row locRows+1)
		dpoint = &img[locCols+2] ;
		cpoint = &img[locCols+2 +1] ;
		MPI_Sendrecv(cpoint,1, grey_coltype, pid, 0,
				dpoint, 1, grey_coltype, pid, 0, comm,MPI_STATUS_IGNORE);

		dpoint = &img[2*(locCols+2) -1];
		cpoint = &img[2*(locCols+2) -2];
		MPI_Sendrecv(cpoint,1, grey_coltype, pid, 0,
				dpoint, 1, grey_coltype, pid, 0, comm,MPI_STATUS_IGNORE);
		//printf(" %d = %d\n", img[2*(locCols+2)],img[2*(locCols+2)+1]);

		// Top and Down row (not including column 0 and column locCols+1)
		dpoint = &img[1];
		cpoint = &img[locCols+2 +1];
		MPI_Sendrecv(cpoint,1, grey_rowtype, pid, 0,
				dpoint, 1, grey_rowtype, pid, 0, comm,MPI_STATUS_IGNORE);

		dpoint = &img[locRows * (locCols+2) +1 ];
		cpoint = &img[(locRows-1)*(locCols+2) +1 ];
		MPI_Sendrecv(cpoint,1, grey_rowtype, pid, 0,
				dpoint, 1, grey_rowtype, pid, 0, comm,MPI_STATUS_IGNORE);

		// Corner pixels
		img[0] = img[1]; 													// left-down pixel
		img[locCols+2 -1] = img[locCols+2 -2];								// right-down pixel
		img[locRows*(locCols+2)] = img[locRows*(locCols+2) +1];				// left-up pixel
		img[(locRows+1)*(locCols+2) -1] = img[(locRows+1)*(locCols+2) -2];	// right-up pixel


/*
		// Left & right most column (not 0 and locRow+2)
		j=1;
		for ( j=1;j<locRows-1;j++){
			img[j*(locCols+2)] = img[j*(locCols+2)+1];
			img[(j+1)*(locCols+2)-1] = img[(j+1)*(locCols+2)-2];
		}
		// Top & down most rows (from 0 to locCol+2)
		j=0;
		for (j=0;j<locCols+2;j++){
			img[0+j] = img[j*(locCols+2)];
			img[(locRows+1)*(locCols+2)+j] = img[(locRows)*(locCols+2)+j];
		}
*/


	} else if (imgType ==  IMG_RGB){

		for (cRow=0; cRow<locRows; cRow++){
			MPI_File_seek(imgf, ( initRow + cRow ) * (3*totalCols) + initCol, MPI_SEEK_SET);
			cpoint = &img[ cRow * ( 3*(locCols+2) ) + 3 ];
			MPI_File_read(imgf, cpoint, locCols, MPI_BYTE, MPI_STATUS_IGNORE);
		}


		// Set the edges to the neighbors values
		// Left and Right column (not including row 0 and row locRows+1)
		dpoint = &img[3*(locCols+2)] ;
		cpoint = &img[3* (locCols+2 +1)] ;
		MPI_Sendrecv(cpoint,1, rgb_coltype, pid, 0,
				dpoint, 1, rgb_coltype, pid, 0, comm,MPI_STATUS_IGNORE);

		dpoint = &img[3*(2*(locCols+2) -1)];
		cpoint = &img[3*(2*(locCols+2) -2)];
		MPI_Sendrecv(cpoint,1, rgb_coltype, pid, 0,
				dpoint, 1, rgb_coltype, pid, 0, comm,MPI_STATUS_IGNORE);
		//printf(" %d = %d\n", img[2*(locCols+2)],img[2*(locCols+2)+1]);

		// Top and Down row (not including column 0 and column locCols+1)
		dpoint = &img[3*1];
		cpoint = &img[3*(locCols+2 +1)];
		MPI_Sendrecv(cpoint,1, rgb_rowtype, pid, 0,
				dpoint, 1, rgb_rowtype, pid, 0, comm,MPI_STATUS_IGNORE);

		dpoint = &img[3*(locRows * (locCols+2) +1) ];
		cpoint = &img[3*((locRows-1)*(locCols+2) +1) ];
		MPI_Sendrecv(cpoint,1, rgb_rowtype, pid, 0,
				dpoint, 1, rgb_rowtype, pid, 0, comm,MPI_STATUS_IGNORE);

		// Corner pixels
		img[3*0] = img[3*1]; 														// left-down pixel
		img[3*(locCols+2 -1)] = img[3*(locCols+2 -2)];								// right-down pixel
		img[3*(locRows*(locCols+2))] = img[3*(locRows*(locCols+2) +1)];				// left-up pixel
		img[3*((locRows+1)*(locCols+2) -1)] = img[3*((locRows+1)*(locCols+2) -2)];	// right-up pixel

		/*		// Set the edges to the neighbors values

		// Left & right most column (not 0 and locRow+2)
		j=1;
		for ( j=1;j<locRows-1;j++){
			img[j*3*(locCols+2)] = img[j*(locCols+2)+1];
			img[(j+1)*(locCols+2)-1] = img[(j+1)*(locCols+2)-2];
		}
		// Top & down most rows (from 0 to locCol+2)
		j=0;
		for (j=0;j<locCols+2;j++){
			img[0+j] = img[j*(locCols+2)];
			img[(locRows+1)*(locCols+2)+j] = img[(locRows)*(locCols+2)+j];
		}*/

	}
	// Close the file
	MPI_File_close(&imgf);
}

void write_full_image(MPI_Comm comm, int pid, char* previmgName, int imgType, char* img,
		int initRow, int initCol, int locRows, int locCols, int totalRows, int totalCols,
		MPI_Datatype grey_rowtype, MPI_Datatype grey_coltype, MPI_Datatype rgb_rowtype,
		MPI_Datatype rgb_coltype){

	int cRow;
	int i,j;
	char* cpoint;
	char* dpoint;
	MPI_File imgf;

	/* Set the new convoluted image file name */
	size_t len = strlen("conv_") + strlen(previmgName) + 1 ;
	char* imgName = malloc(len);
	sprintf(imgName, "conv_%s", previmgName);

	// Open file for parallel reading
	MPI_File_open(comm, imgName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &imgf);

	if (imgType == IMG_GREY){

		for ( cRow = 0; cRow<locRows; cRow++){
			MPI_File_seek(imgf, (initRow + cRow) * totalCols + initCol, MPI_SEEK_SET);
			cpoint = &img[ cRow * (locCols+2) + 1 ];
			MPI_File_write(imgf, cpoint, locCols, MPI_BYTE, MPI_STATUS_IGNORE);
		}

	} else if (imgType ==  IMG_RGB){

		for (cRow=0; cRow<locRows; cRow++){
			MPI_File_seek(imgf, ( initRow + cRow ) * (3*totalCols) + initCol, MPI_SEEK_SET);
			cpoint = &img[ cRow * ( 3*(locCols+2) ) + 3 ];
			MPI_File_write(imgf, cpoint, locCols, MPI_BYTE, MPI_STATUS_IGNORE);
		}

	}
	// Close the file
	MPI_File_close(&imgf);
}

int allocate_img_arrays(int locRows, int locCols, int imgType, uint8_t** imgcurr, uint8_t** imgnext){

	int halo;

	/* Set the ghost area around columns */
	if (imgType == IMG_GREY) { halo = 1;} else if (imgType == IMG_RGB) { halo = 3; }

	/* Allocate appropriate local image array size : (locRows + 2) x (locCols+2) * ghost */
	*imgcurr = calloc((locRows+2)*(locCols+2)*halo, sizeof(uint8_t));
	*imgnext = calloc((locRows+2)*(locCols+2)*halo, sizeof(uint8_t));

	/* Return failure if memory allocation failed*/
	if(*imgcurr==NULL || *imgnext==NULL){
		return 0;
	}

	/* Return success */
	return 1;

}

void get_parameters(int argc, char** argv, char** imgName, int* numRows, int* numCols,
		int* maxIters, int* convIters, img_color* imgType) {

	if ( argc == 7 && ( strcmp(argv[6],"grey")==0 || strcmp(argv[6],"rgb") == 0 )){
		// Get image name
		*imgName = malloc( ( strlen(argv[1])+1 ) * sizeof(char) );
		*imgName = argv[1];
		// Get number of image rows
		*numRows = atoi(argv[2]);
		// Get number of image columns
		*numCols = atoi(argv[3]);
		// Get number of max iterations
		*maxIters = atoi(argv[4]);
		// Get number of convergence check iterations
		*convIters = atoi(argv[5]);
		// Get image type (grey/rgb)
		if ( strcmp(argv[6],"grey")==0 ) {
			*imgType = IMG_GREY;
		} else {
			*imgType = IMG_RGB;
		}
	} else {
		fprintf(stderr, "Input error : Use of program should be :\n");
		fprintf(stderr,"%s [imageName] [imagePixelRows] [imagePixelColumns] [maxIterations] [convergenceCheckIterations] [grey/rgb]\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(EXIT_FAILURE);
	}

}

int columns_splits(int nrows, int ncols, int nprocs){
	int csplits, csplits_ceil, csplits_floor;
	int rsplits, rsplits_ceil, rsplits_floor;
	int tmpPerimeter, minPerimeter;
	float tmp;
	int tmpceil;
	int tmpfloor;
	int res =-1;
	int perim = nrows*ncols, perim_ceil = nrows*ncols, perim_floor = nrows*ncols;


	/************ NUMERICAL APPROACH **************/

	/* Assign a very high (not real) perimeter for checking against minimum found in every iteration */
	minPerimeter = nprocs*nrows + nprocs*ncols;

	/* Check every feasible value of column splits starting from 1 up to number of processes */
	for(csplits = 1 ; csplits < nprocs+1; csplits++){

		/* Checking if number of columns divided by current column splits gives no residual */
		if (ncols % csplits == 0){

			/* Checking if number of total splits (nprocs) divided by current column splits gives no residual */
			if ( nprocs % csplits == 0 ) {

				/* Calculate number of row splits (should be integer) */
				rsplits = nprocs/csplits;

				/* Checking if number of row splits divides exactly the number of rows */
				if (nrows % rsplits == 0) {

					/* Calculate the perimeter */
					tmpPerimeter = (csplits-1)*nrows + (rsplits-1)*ncols;

					/* Compare against minimum so far*/
					if ( tmpPerimeter < minPerimeter ) {
						/* If better (smaller) set it as minimum and keep result*/
						minPerimeter = tmpPerimeter;
						res = csplits;
					}
				}
			}

		}
	}

	return res;

/*	***** THEORETICAL APPROACH *****

	// Calc the theoretical best split
	tmp = sqrt((float)ncols * (float)nprocs / ( (float) nrows) );//2,29

	// Calc theoretical best split ceil
	tmpceil = (int) ceil(tmp); //3
	printf("tmpceil = %d\n", tmpceil);
	// If column splits ceil divides number of columns proceed
	if ( ncols % tmpceil == 0 ){
		// If row splits calculated divides number of rows proceed
		if ( ( nprocs % tmpceil == 0 ) && ( nrows % (nprocs/tmpceil) == 0 )  ) {
			csplits_ceil = tmpceil;
			rsplits_ceil = nprocs/tmpceil;
			// Calculate perimeter for ceil
			perim_ceil = nrows/rsplits_ceil + ncols/csplits_ceil;
			printf("perim_ceil  = %d\n", perim_ceil);
			res = 1;
		}
	}

	// Calc theoretical best split floor
	tmpfloor = floor(tmp);
	printf("tmpfloor = %d\n", tmpfloor); //2
	// If column splits floor divides number of columns proceed
	if ( ncols % tmpfloor == 0  ){
		// If row splits calculated divides number of rows proceed
		if ( ( nprocs % tmpfloor == 0 ) && ( nrows % (nprocs/tmpfloor) == 0 )  ) {
			csplits_floor = tmpfloor;
			rsplits_floor = nprocs/tmpfloor;
			// Calculate perimeter for floor
			perim_floor = nrows/rsplits_floor + ncols/csplits_floor;
			printf("perim_floor  = %d\n", perim_floor);
			res = 1;
		}
	}

	// If neither of ceil/floor values divide rows/columns return false
	if (res == -1) return res;

	// Return least perimeter value
	if ( perim_ceil <= perim_floor ){
		csplits = csplits_ceil;
	} else {
		csplits = csplits_floor;
	}

	return csplits;*/

}
