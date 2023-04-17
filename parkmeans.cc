/**
 * Paraller k-means clustering algoritm implementation with MPI
 * @author Lukáš Plevač <xpleva07@vut.cz>
 * @date 2023.04.17
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#include <fstream>
#include <limits>
#include <string.h>

#define INPUT_FILE "numbers"
#define MPI_ROOT_RANK     0
#define NEW               1
#define OLD               0
#define CENTEROIDS_COUNT  4

//#define DEBUG
#if defined(DEBUG)
    #define DEBUG_PRINT(fmt, args...) fprintf(stderr, "[RANK: %d][DEBUG] %s:%d:%s(): " fmt, m_rank, __FILE__, __LINE__, __func__, ##args)
#else
    #define DEBUG_PRINT(fmt, args...) {}
#endif

/**
 * Load uint number from binary file INPUT_FILE to std::vecor
 * @return std::vector<uint8_t> of bytes in file
 */
std::vector<uint8_t> loadData(size_t readSize) {
    std::vector<uint8_t> fileBuffer;

    try {
        std::ifstream ifs(INPUT_FILE, std::ios_base::binary);

        // get length of file
        ifs.seekg(0, ifs.end);
        size_t length = ifs.tellg();
        ifs.seekg(0, ifs.beg);

        // read max of read size
        length = length < readSize ? length : readSize;
        
        //read file
        if (length > 0) {
            fileBuffer.resize(length);    
            ifs.read((char*) &fileBuffer[0], length);
        }
    } catch (...) {
        fprintf(stderr, "Fail to load input file\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return fileBuffer;
}

/**
 * Print points in one cluster by centeroid
 * @param centeroids     pointer to float array of centeroids
 * @param centeroidIndex index of centeroid of cluter to print
 * @param array          vector of uint8_t numbers in all clusters 
 */
void printSubPoints(float *centeroids, size_t centeroidIndex, std::vector<uint8_t> &array) {
    
    std::string line = "";

    for (size_t p = 0; p < array.size(); p++) {
        size_t nearestCemteroid = 0;
        float  nearestDistance  = std::numeric_limits<float>::infinity();

        // compute nearist centeroid
        for (size_t i = 0; i < CENTEROIDS_COUNT; i++) {
            float thisDistance = std::abs(centeroids[i] - array[p]);

            if (thisDistance < nearestDistance) { // find nearist centeroids in old centeroids array
                nearestDistance  = thisDistance;
                nearestCemteroid = i;
            }
        }

        if (nearestCemteroid == centeroidIndex) line += " " + std::to_string(array[p]) + ",";
    }

    if (line.length() > 0) line.pop_back(); // remove last ,

    printf("%s\n", line.c_str());
}

int main(int argc, char** argv) {
    int m_rank, m_rankSize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_rankSize);

    std::vector<uint8_t> fileBuffer;
    unsigned             fileBufferSize = 0;

    fileBuffer.reserve(CENTEROIDS_COUNT);
    if (m_rank == MPI_ROOT_RANK) {
        fileBuffer = loadData(m_rankSize);

        if (fileBuffer.size() < m_rankSize) {
            fprintf(stderr, "Input file is too small\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    uint8_t localNumber;
    float   globalCenteroids1[CENTEROIDS_COUNT * 2] = {float(fileBuffer[0]), float(fileBuffer[1]), float(fileBuffer[2]), float(fileBuffer[3])};
    float   globalCenteroids2[CENTEROIDS_COUNT * 2];
    float   localCenteroids  [CENTEROIDS_COUNT * 2];

    float  *workCenteroids[2]    = {globalCenteroids1, globalCenteroids2};

    // scatter numbers over all ranks
    MPI_Scatter(fileBuffer.data(), 1, MPI_UINT8_T, &localNumber, 1, MPI_UINT8_T, MPI_ROOT_RANK, MPI_COMM_WORLD);
    
    // broadcast init centeroids to all
    MPI_Bcast(globalCenteroids1, CENTEROIDS_COUNT, MPI_FLOAT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    DEBUG_PRINT("Scatter done centers are %f %f %f %f number is %d\n", globalCenteroids1[0], globalCenteroids1[1], globalCenteroids1[2], globalCenteroids1[3], localNumber);

    // k-means loop until centeroids not change
    while (true) {
        // compute new centeroids
        size_t nearestCemteroid = 0;
        float  nearestDistance  = std::numeric_limits<float>::infinity();

        // compute nearist centeroid
        for (size_t i = 0; i < CENTEROIDS_COUNT; i++) {
            float thisDistance = std::abs(localNumber - workCenteroids[OLD][i]);

            if (thisDistance < nearestDistance) { // find nearist centeroids in old centeroids array
                nearestDistance  = thisDistance;
                nearestCemteroid = i;
            }

            localCenteroids[i]                    = 0; // clear new centeroids
            localCenteroids[i + CENTEROIDS_COUNT] = 0; // clear sums
        }

        // Add self number to nearest centeroid
        localCenteroids[nearestCemteroid]                    = float(localNumber);
        localCenteroids[nearestCemteroid + CENTEROIDS_COUNT] = 1;

        // reduce sum for new centeroids and clout of numbers in cluster
        MPI_Allreduce(localCenteroids, workCenteroids[NEW], CENTEROIDS_COUNT * 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        DEBUG_PRINT("New centers sums are %f %f %f %f\n", workCenteroids[NEW][0], workCenteroids[NEW][1], workCenteroids[NEW][2], workCenteroids[NEW][3]);
        DEBUG_PRINT("Centers counts are %f %f %f %f\n", workCenteroids[NEW][4], workCenteroids[NEW][5], workCenteroids[NEW][6], workCenteroids[NEW][7]);

        // compute mean on all and check if its same
        bool timeToEnd = true;
        for (size_t i = 0; i < CENTEROIDS_COUNT; i++) {
            workCenteroids[NEW][i] = (workCenteroids[NEW][i + CENTEROIDS_COUNT] == 0 ? workCenteroids[OLD][i] : workCenteroids[NEW][i] / workCenteroids[NEW][i + CENTEROIDS_COUNT]);

            if (std::abs(workCenteroids[NEW][i] - workCenteroids[OLD][i]) > 0.01) timeToEnd = false;
        }

        // check if its same
        if (timeToEnd) break;

        // exchage old and new array
        std::swap(workCenteroids[NEW], workCenteroids[OLD]);
    }

    // last print data with centeroids
    if (m_rank == MPI_ROOT_RANK) {
        for (size_t i = 0; i < CENTEROIDS_COUNT; i++) {
            printf("[%.1f]", workCenteroids[NEW][i]);
            printSubPoints(workCenteroids[NEW], i, fileBuffer);
        }
    }
        

    MPI_Finalize();
    return 0;
}