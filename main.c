#include<stdio.h>
#include<stdlib.h>

#define inputSize 16
#define cellNumInLayerA 64
#define cellNumInLayerB 128
#define cellNumInLayerC 128
#define cellNumInLayerD 4
#define maxCellNum 128
#define learnRate 0.00000001

double *paramForLayerA = 0;
double *paramForLayerB = 0;
double *paramForLayerC = 0;
double *paramForLayerD = 0;
double *bufferA = 0;
double *bufferB = 0;
double *gradForLayerA = 0;
double *gradForLayerB = 0;
double *gradForLayerC = 0;
double *gradForLayerD = 0;

void flushDoubleMem(double *p, int num);
void fillRandomMem(double *p, int num);
void initMem();
void flushGrad();
void inference(double *x);
void loss(double *target);
void backward();
void updateParam();
void dumpDouble(double *p, int num);

int main()
{
	double x[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6};
	double target[] = {0.1234, 0.5678, 0.9011, 0.1112};	
	initMem();
	for(int i = 0; i < 128; i++)
	{
		flushGrad();	
		inference(x);
		loss(target);
		backward();
		updateParam();
	}
}

void flushDoubleMem(double *p, int num)
{
	for(int i = 0; i < num; i++)
		p[i] = 0.0;
}

void fillRandomMem(double *p, int num)
{
	for(int i = 0; i < num; i++)
                p[i] = (rand() % 1024) / 65536.0;
}

void initMem()
{
	paramForLayerA = (double *)malloc(cellNumInLayerA * (inputSize + 1) * sizeof(double));
	paramForLayerB = (double *)malloc(cellNumInLayerB * (cellNumInLayerA + 1) * sizeof(double));
	paramForLayerC = (double *)malloc(cellNumInLayerC * (cellNumInLayerB + 1) * sizeof(double));
	paramForLayerD = (double *)malloc(cellNumInLayerD * (cellNumInLayerC + 1) * sizeof(double));
	bufferA = (double *)malloc(maxCellNum * sizeof(double));
	bufferB = (double *)malloc(maxCellNum * sizeof(double));
	gradForLayerA = (double *)malloc(cellNumInLayerA * sizeof(double));
	gradForLayerB = (double *)malloc(cellNumInLayerB * sizeof(double));
	gradForLayerC = (double *)malloc(cellNumInLayerC * sizeof(double));
	gradForLayerD = (double *)malloc(cellNumInLayerD * sizeof(double));
}

void flushGrad()
{
	flushDoubleMem(gradForLayerA, cellNumInLayerA);
        flushDoubleMem(gradForLayerB, cellNumInLayerB);
        flushDoubleMem(gradForLayerC, cellNumInLayerC);
        flushDoubleMem(gradForLayerD, cellNumInLayerD);
}

void inference(double *x)
{
	flushDoubleMem(bufferA, maxCellNum);

	for(int i = 0; i < cellNumInLayerA; i++)
	{
		for(int j = 0; j < inputSize; j++)
			bufferA[i] += paramForLayerA[i * (inputSize + 1) + j] * x[j];
		bufferA[i] += paramForLayerA[i * (inputSize + 1) + inputSize];
	}
	//printf("Now Dump for LayerA\n");
	//dumpDouble(bufferA, cellNumInLayerA);
	
	flushDoubleMem(bufferB, maxCellNum);

	for(int i = 0; i < cellNumInLayerB; i++)
	{
		for(int j = 0; j < cellNumInLayerA; j++)
			bufferB[i] += paramForLayerB[i * (cellNumInLayerA + 1) + j] * bufferA[j];
		bufferB[i] += paramForLayerB[i * (cellNumInLayerA + 1) + cellNumInLayerA];
	}
	//printf("Now Dump for LayerB\n");
	//dumpDouble(bufferB, cellNumInLayerB);

	flushDoubleMem(bufferA, maxCellNum);

	for(int i = 0; i < cellNumInLayerC; i++)
	{
		for(int j = 0; j < cellNumInLayerB; j++)
			bufferA[i] += paramForLayerC[i * (cellNumInLayerB + 1) + j] * bufferB[j];
		bufferA[i] += paramForLayerC[i * (cellNumInLayerB + 1) + cellNumInLayerB];
	}
	//printf("Now Dump for LayerC\n");
	//dumpDouble(bufferA, cellNumInLayerC);
	
	flushDoubleMem(bufferB, maxCellNum);

	for(int i = 0; i < cellNumInLayerD; i++)
	{
		for(int j = 0; j < cellNumInLayerC; j++)
			bufferB[i] += paramForLayerD[i * (cellNumInLayerC + 1) + j] * bufferA[j];
		bufferB[i] += paramForLayerD[i * (cellNumInLayerC + 1) + cellNumInLayerC];
	}
	printf("Now Dump for LayerD\n");
	dumpDouble(bufferB, cellNumInLayerD);

	flushDoubleMem(bufferA, maxCellNum);
	
	double *exchange = bufferA;
	bufferA = bufferB;
	bufferB = exchange;
}

void loss(double *target)
{
	for(int i = 0; i < cellNumInLayerD; i++)
		bufferB[i] = bufferA[i] - target[i];
	printf("Now Dump for Loss\n");
	dumpDouble(bufferB, cellNumInLayerD);

	flushDoubleMem(bufferA, cellNumInLayerD);

	double *exchange = bufferA;
        bufferA = bufferB;
        bufferB = exchange;
}

void backward()
{
	for(int i = 0; i < cellNumInLayerD; i++)
		gradForLayerD[i] = bufferA[i];
	
	for(int i = 0; i < cellNumInLayerD; i++)
		for(int j = 0; j < cellNumInLayerC; j++)
			gradForLayerC[j] += gradForLayerD[i] * paramForLayerD[i * (cellNumInLayerC + 1) + j];

	for(int i = 0; i < cellNumInLayerC; i++)
		for(int j = 0; j < cellNumInLayerB; j++)
			gradForLayerB[j] += gradForLayerC[i] * paramForLayerC[i * (cellNumInLayerB + 1) + j];
	
	for(int i = 0; i < cellNumInLayerB; i++)
		for(int j = 0; j < cellNumInLayerA; j++)
			gradForLayerA[j] += gradForLayerB[i] * paramForLayerC[i * (cellNumInLayerA + 1) + j];

}

void backward2()
{
	for(int i = 0; i < cellNumInLayerD; i++)
                gradForLayerD[i] = bufferA[i];
	
	for(int i = 0; i < cellNumInLayerC; i++)
		for(int j = 0; j < cellNumInLayerD; j++)
			gradForLayerC[i] += paramForLayerD[j * (cellNumInLayerC + 1) + i] * gradForLayerD[j];
	
	for(int i = 0; i < cellNumInLayerB; i++)
                for(int j = 0; j < cellNumInLayerC; j++)
			gradForLayerB[i] += paramForLayerC[j * (cellNumInLayerB + 1) + i] * gradForLayerC[j];
	
	for(int i = 0; i < cellNumInLayerA; i++)
		for(int j = 0; j < cellNumInLayerB; j++)
			gradForLayerA[i] += paramForLayerB[j * (cellNumInLayerA + 1) + i] * gradForLayerB[j];
}

void updateParam()
{
	float temp = 0.0;
	for(int i = 0; i < cellNumInLayerD; i++)
	{
		for(int j = 0; j < cellNumInLayerC; j++)
		{
			temp = learnRate * paramForLayerD[i * (cellNumInLayerC + 1) + j] * gradForLayerD[i];
			paramForLayerD[i * (cellNumInLayerC + 1) + j] -= temp;
		}
		paramForLayerD[i * (cellNumInLayerC + 1) + cellNumInLayerC] -= gradForLayerD[i];
	}

	for(int i = 0; i < cellNumInLayerC; i++)
	{
		for(int j = 0; j < cellNumInLayerB; j++)
		{
                        temp = learnRate * paramForLayerC[i * (cellNumInLayerB + 1) + j] * gradForLayerC[i];
                        paramForLayerC[i * (cellNumInLayerB + 1) + j] -= temp;
                }
		paramForLayerC[i * (cellNumInLayerB + 1) + cellNumInLayerB] -= gradForLayerC[i];
	}

	for(int i = 0; i < cellNumInLayerB; i++)
        {
                for(int j = 0; j < cellNumInLayerA; j++)
                {
                        temp = learnRate * paramForLayerB[i * (cellNumInLayerA + 1) + j] * gradForLayerB[i];
                        paramForLayerB[i * (cellNumInLayerA + 1) + j] -= temp;
                }
                paramForLayerB[i * (cellNumInLayerA + 1) + cellNumInLayerA] -= gradForLayerB[i];
        }

	for(int i = 0; i < cellNumInLayerA; i++)
        {
                for(int j = 0; j < inputSize; j++)
                {
                        temp = learnRate * paramForLayerB[i * (inputSize + 1) + j] * gradForLayerA[i];
                        paramForLayerA[i * (inputSize + 1) + j] -= temp;
                }
                paramForLayerA[i * (inputSize + 1) + inputSize] -= gradForLayerA[i];
        }
}

void dumpDouble(double *p, int num)
{
	for(int i = 0; i < num; i++)
		printf("%.4lf, ", p[i]);
	printf("\n");	
}
