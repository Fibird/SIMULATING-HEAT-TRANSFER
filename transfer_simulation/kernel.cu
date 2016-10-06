#include "cuda.h"
#include "cpu_anim.h"
#include "book.h"
#include <stdio.h>

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

texture<float> texConstSrc;
texture<float> texIn;
texture<float> texOut;

__global__ void copy_const_kernel(float *iptr)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if (c != 0)
		iptr[offset] = c;
}

__global__ void blend_kernel(float *dst, bool distOut)
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.x * blockIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)
		left++;
	if (x == DIM - 1)
		right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0)
		top += DIM;
	if (y == DIM - 1)
		bottom -= DIM;

	float t, l, c, r, b;
	if (distOut)
	{
		t = tex1Dfetch(texIn, top);
		l = tex1Dfetch(texIn, left);
		c= tex1Dfetch(texIn, offset);
		r = tex1Dfetch(texIn, right);
		b = tex1Dfetch(texIn, bottom);
	}
	else
	{
		t = tex1Dfetch(texOut, top);
		l = tex1Dfetch(texOut, left);
		c = tex1Dfetch(texOut, offset);
		r = tex1Dfetch(texOut, right);
		b = tex1Dfetch(texOut, bottom);
	}
	dst[offset] = c + SPEED * (t + b + l + r - 4 * c);
}

// global needed by the update routine
struct DataBlock
{
	unsigned char *output_bitmap;
	float *dev_inSrc;
	float *dev_outSrc;
	float *dev_constSrc;
	CPUAnimBitmap *bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

void anim_gpu(DataBlock *d, int ticks)
{
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	CPUAnimBitmap *bitmap = d->bitmap;

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i = 0; i < 90; i++)
	{
		float *in, *out;
		if (dstOut)
		{
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		}
		else
		{
			out = d->dev_inSrc;
			in = d->dev_outSrc;
		}
		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>>(out, dstOut);
		dstOut = !dstOut;
	}
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);
	cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost);
	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);

	d->totalTime += elapsedTime;
	++d->frames;
	printf("Average Time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock *d)
{
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);
	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());

	// assume float == 4 chars in size(i.e, rgba)
	cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());
	cudaBindTexture(NULL, texConstSrc, data.dev_constSrc, bitmap.image_size());
	cudaBindTexture(NULL, texIn, data.dev_inSrc, bitmap.image_size());
	cudaBindTexture(NULL, texOut, data.dev_outSrc, bitmap.image_size());

	float *temp = (float*)malloc(bitmap.image_size());
	for (int i = 0; i < DIM * DIM; i++)
	{
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x>300) && (x<600) && (y>310) && (y<601))
			temp[i] = MAX_TEMP;
	}
	temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
	temp[DIM*700+100] = MIN_TEMP;
	temp[DIM*300+300] = MIN_TEMP;
	temp[DIM*200+700] = MIN_TEMP;
	for (int y=800; y<900; y++) 
	{
		for (int x=400; x<500; x++) 
		{
			temp[x+y*DIM] = MIN_TEMP;
		}
	}
	cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
	for (int y = 800; y < DIM; y++)
	{
		for (int x = 0; x < 200; x++)
		{
			temp[x + y * DIM] = MAX_TEMP;
		}
	}
	cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
	free(temp);
	bitmap.anim_and_exit((void (*)(void *, int))anim_gpu, (void (*)(void*))anim_exit);
}