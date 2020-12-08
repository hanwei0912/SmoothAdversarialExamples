#include "math.h"
#include "mex.h"   

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{    

    mxArray *xData,*yData;
    double *xVal, *yVal, *outStd, *outStdMod, *outMean;
    double mean, mean2, stdev;
    double *TMP;
    int i,j,iB,jB;
    int rowLen, colLen;


    xData = prhs[0];
    yData = prhs[1];


    xVal = mxGetPr(xData);
    rowLen  = mxGetN(xData);
    colLen  = mxGetM(xData);
    

    yVal = mxGetPr(yData);
    rowLen  = mxGetN(yData);
    colLen  = mxGetM(yData);


    plhs[0] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); 
    outStd  = mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL); 
    outStdMod  = mxGetPr(plhs[1]);

    plhs[2] = mxCreateDoubleMatrix(colLen, rowLen, mxREAL);
    outMean  = mxGetPr(plhs[2]);
      
    TMP = mxGetPr( mxCreateDoubleMatrix(colLen, rowLen, mxREAL) );


    for(i=0; i<rowLen-15; i += 4)
    {
        for(j=0; j<colLen-15; j += 4)
        {

            mean = 0;
            mean2= 0;
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     mean +=  xVal[(iB*colLen)+jB];
                     mean2 += yVal[(iB*colLen)+jB];
                               
                }
            }            
            mean = mean / 256.0;
            mean2= mean2 / 256.0;
            
            stdev = 0;            
            for( iB = i; iB < i+16; iB++ )
            {
                for( jB = j; jB < j+16; jB++ )
                {
                     stdev += pow( xVal[(iB*colLen)+jB]-mean, 2.0 );                     
                }
            }                        
            stdev = sqrt(stdev / 255.0);                     
            
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {
                    outMean[(iB*colLen)+jB] = mean2;
                    outStd[(iB*colLen)+jB]  = stdev;
                }
            }                       
        }
    }
    
 
        
    for(i=0; i<rowLen-15; i += 4)
    {
        for(j=0; j<colLen-15; j += 4)
        {

            mean = 0;
            
            for( iB = i; iB < i+8; iB ++ )
            {
                for( jB = j; jB < j+8; jB ++ )
                {
                     mean +=  yVal[(iB*colLen)+jB];                                                    
                }
            }            
            mean = mean / 64.0;            
            
           
            stdev = 0;            
            for( iB = i; iB < i+8; iB++ )
            {
                for( jB = j; jB < j+8; jB++ )
                {
                     stdev += pow( yVal[(iB*colLen)+jB]-mean, 2.0 );                     
                }
            }                        
            stdev = sqrt(stdev / 63.0);                    
            
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {                    
                    TMP[(iB*colLen)+jB]  = stdev;
                    outStdMod[(iB*colLen)+jB] = stdev;
                }
            }                       
        }
    }
    
    for(i=0; i<rowLen-15; i += 4)
    {        
        for(j=0; j<colLen-15; j += 4)
        {
            mean = TMP[(i*colLen)+j];
            for( iB = i; iB < i+8; iB += 5 )
            {
                for( jB = j; jB < j+8; jB += 5 )
                {
                   if( iB < rowLen-15 && jB < colLen-15 && mean > TMP[(iB*colLen)+jB]  )
                       mean = TMP[(iB*colLen)+jB];
                }
            }
                       
            for( iB = i; iB < i+4; iB++ )
            {
                for( jB = j; jB < j+4; jB++ )
                {
                     outStdMod[(iB*colLen)+jB] = mean;                                                   
                }
            }     
        }
    }
    mxDestroyArray(TMP);

    return;
}
        