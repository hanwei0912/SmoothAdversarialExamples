#include "math.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    

    mxArray *xData;
    double *xVal, *outStd, *outSkw, *outKrt, *outMean;
    double mean, stdev, skw, krt, stmp;
    int i,j,iB,jB;
    int rowLen, colLen;

    xData = prhs[0];

    xVal    = mxGetPr(xData);
    rowLen  = mxGetN(xData);
    colLen  = mxGetM(xData);

    plhs[0] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL);
    outStd  = mxGetPr(plhs[0]);
    

    plhs[1] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL);   
    outSkw  = mxGetPr(plhs[1]);
    

    plhs[2] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL);  
    outKrt  = mxGetPr(plhs[2]);
        

    for(i=0; i<rowLen-15; i += 4)
    {
        for(j=0; j<colLen-15; j += 4)
        {

            mean = 0;
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     mean += xVal[(iB*colLen)+jB];
                               
                }
            }            
            mean = mean / 256.0;
            
            stdev = 0;
            skw   = 0;
            krt   = 0;
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     stdev += pow( xVal[(iB*colLen)+jB]-mean, 2.0 );
                     skw   += pow( xVal[(iB*colLen)+jB]-mean, 3.0 );
                     krt   += pow( xVal[(iB*colLen)+jB]-mean, 4.0 );
                }
            }            
            stmp  = sqrt(stdev / 256.0);
            stdev = sqrt(stdev / 255.0);
            
            if( stmp != 0 ){
                skw   = (skw/256.0) / pow( stmp, 3 );
                krt   = (krt/256.0) / pow( stmp, 4 );
            }
            else{
                skw = 0;
                krt = 0;
            }
            
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {                    
                    outStd[(iB*colLen)+jB]  = stdev;
                    outSkw[(iB*colLen)+jB]  = skw;
                    outKrt[(iB*colLen)+jB]  = krt;            
                }
            }
        }
    }


    return;
}
        