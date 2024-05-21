#include <stdio.h>
#include <stdint.h>

extern "C"
{
    __declspec(dllexport) void contact(int, float, int, int, uint8_t **, uint8_t **, uint8_t **, uint8_t **, float **, uint8_t **, uint8_t **);
}

uint8_t close(float a, float b, float threshold)
{
    float x = a-b;
    if(x < 0)
    {
        x = -x;
    }
    return (x<threshold);
}

void contactDM(int searchWidth, float threshold, int rows, int cols, uint8_t **dil, uint8_t **maskN, uint8_t **maskD, float **depth, uint8_t **contP)
{
    float **crdepth = depth;
    uint8_t **crdil=dil;
    uint8_t **crmaskN=maskN;
    uint8_t **crmaskD=maskD;
    uint8_t **crcontP=contP;
    int lastSearch = searchWidth+1;
    for(int i=0; i < rows; i++)
    {
        float *cdepth = *crdepth;
        uint8_t *cdil=*crdil;
        uint8_t *cmaskN=*crmaskN;
        //uint8_t *cmaskD=*crmaskD;
        uint8_t *ccontP=*crcontP;
        for(int j=0; j < cols; j++)
        {
            if((0!=*cdil) && (0!=*cmaskN))
            {
                uint8_t found = 0;
                int k = -searchWidth;
                float **cordepth = crdepth+k;
                uint8_t **cormaskD = crmaskD+k;
                while((0==found) && (k < lastSearch))
                {
                    int l = -searchWidth;
                    float *codepth = (*cordepth)+j+l;
                    uint8_t *comaskD = (*cormaskD)+j+l;
                    while((0==found) && (l < lastSearch))
                    {
                        int cr = i+k;
                        int cc = j+l;
                        if((0<=cr) && (cr<rows) && (0<=cc) && (cc<cols) && (0!=*comaskD) && close(*cdepth, *codepth, threshold))
                        {
                            found = 255;
                        }
                        l++;
                        codepth++;
                        comaskD++;
                    }
                    k++;
                    cordepth++;
                    cormaskD++;
                }
                if(0 != found)
                {
                    *ccontP = 255;
                }
            }
            cdepth++;
            cdil++;
            cmaskN++;
            //cmaskD++;
            ccontP++;
        }
        crdepth++;
        crdil++;
        crmaskN++;
        crmaskD++;
        crcontP++;
    }
}

void contact(int searchWidth, float threshold, int rows, int cols, uint8_t **dilS, uint8_t **dilO, uint8_t **maskS, uint8_t **maskO, float **depthImg, uint8_t **contPS, uint8_t **contPO)
{
    contactDM(searchWidth, threshold, rows, cols, dilS, maskO, maskS, depthImg, contPO);
    contactDM(searchWidth, threshold, rows, cols, dilO, maskS, maskO, depthImg, contPS);
}