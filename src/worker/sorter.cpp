#include <emscripten/emscripten.h>
#include <iostream>

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN
#endif

EXTERN EMSCRIPTEN_KEEPALIVE void sortIndexes(unsigned int* indexes, int* positions, char* sortBuffers, int* viewProj,
                                             unsigned int* indexesOut, float cameraX, float cameraY,
                                             float cameraZ, unsigned int distanceMapRange, unsigned int sortCount, unsigned int vertexCount) {

    int maxDistance = -2147483648;
    int minDistance = 2147483647;
    int* distances = (int*)sortBuffers;
    for (unsigned int i = 0; i < sortCount; i++) {
        unsigned int indexOffset = 3 * (unsigned int)indexes[i];
        int depth =
            (int)((viewProj[2] * positions[indexOffset] +
                   viewProj[6] * positions[indexOffset + 1] +
                   viewProj[10] * positions[indexOffset + 2]));
        distances[i] = depth;
        if (depth > maxDistance) maxDistance = depth;
        if (depth < minDistance) minDistance = depth;
    }

    float distancesRange = (float)maxDistance - (float)minDistance;
    float rangeMap = (float)distanceMapRange / distancesRange;

    unsigned int* frequencies = ((unsigned int *)distances) + vertexCount;
    unsigned int* realIndex = ((unsigned int *)frequencies) + distanceMapRange;

    for (unsigned int i = 0; i < sortCount; i++) {
        unsigned int frequenciesIndex = (int)((float)(distances[i] - minDistance) * rangeMap);
        unsigned int cFreq = frequencies[frequenciesIndex];
        frequencies[frequenciesIndex] = cFreq + 1;   
    }

    unsigned int cumulativeFreq = 0;
    for (unsigned int i = 1; i < distanceMapRange; i++) {
        unsigned int cFreq = frequencies[i];
        cumulativeFreq += cFreq;
        frequencies[i] = cumulativeFreq;
    }

    for (int i = sortCount - 1; i >= 0; i--) {
        unsigned int frequenciesIndex =  (int)((float)(distances[i] - minDistance) * rangeMap);
        unsigned int freq = frequencies[frequenciesIndex];
        realIndex[freq - 1] = indexes[i];
        frequencies[frequenciesIndex] = freq - 1;
    }

}