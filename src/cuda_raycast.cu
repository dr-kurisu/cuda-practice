#include <cuda_runtime.h>
#include <cstdio>

struct Sphere {
    float3 centre;
    float radius;
};

struct Ray {
    float3 origin;
    float3 direction;
};


struct Camera {
    float3 position;

    float planeLeft;
    float planeRight;
    float planeBottom;
    float planeTop;
    float planeZ;
};

__host__ __device__
float3 operator-(const float3& a, const float3& b)
{
    return make_float3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

__host__ __device__
float dot(const float3& a, const float3& b)
{
    return a.x * b.x +
           a.y * b.y +
           a.z * b.z;
}

__host__ __device__
float3 normalise(const float3& v)
{
    float length = sqrtf(
        v.x * v.x +
        v.y * v.y +
        v.z * v.z
    );

    return make_float3(
        v.x / length,
        v.y / length,
        v.z / length
    );
}


__device__
bool intersect(const Ray& ray, const Sphere& sphere)
{
    float3 oc = ray.origin - sphere.centre;

    float b = dot(ray.direction, oc);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float discriminant = b * b - c;

    return discriminant >= 0.0f;
}

__global__
void raycast(unsigned char* image, const Sphere* spheres, int spheresCount, int width, int height, const Camera camera) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Convert (x, y) to a point on the projection plane
    float px = camera.planeLeft + (x + 0.5f) / width * (camera.planeRight - camera.planeLeft);
    float py = camera.planeBottom + (y + 0.5f) / height * (camera.planeTop - camera.planeBottom);

    // Construct ray from camera to that point
    float3 pointOnPlane = {px, py, camera.planeZ};
    float3 direction = pointOnPlane - camera.position;
    direction = normalise(direction);

    Ray ray = {
        camera.position,
        direction
    };

    bool intersected = false;
    // Test ray against every sphere
    for (int i = 0; i < spheresCount; ++i) {
        if (intersect(ray, spheres[i])) {
            intersected = true;
        }
    }

    // Write result to image
    if (intersected) {
        image[y * width + x] = 255;
    }else {
        image[y * width + x] = 0;
    }
}

void writeImage(const unsigned char* images, int width, int height) {
    FILE* file = fopen("output.ppm", "wb");

    fprintf(file, "P5\n%d %d\n255\n", width, height);

    fwrite(
        images,
        sizeof(unsigned char),
        width * height,
        file
    );

    fclose(file);

}



int main() {
    Camera camera = {
        {0.0f, 0.0f, 0.0f},  // position

        -2.0f,                // planeLeft
         2.0f,                // planeRight
        -1.5f,                // planeBottom
         1.5f,                // planeTop
         1.0f                 // planeZ
    };
    int spheresCount = 20;

    Sphere spheres[] = {
        {{ 0.0f,  0.0f,  5.0f}, 1.0f},
        {{-1.5f,  0.0f,  6.0f}, 1.0f},
        {{ 1.5f,  0.0f,  6.0f}, 1.0f},

        {{-3.0f,  1.5f,  7.0f}, 0.8f},
        {{ 0.0f,  1.5f,  7.0f}, 0.8f},
        {{ 3.0f,  1.5f,  7.0f}, 0.8f},

        {{-3.0f, -1.5f, 7.0f}, 0.8f},
        {{ 0.0f, -1.5f, 7.0f}, 0.8f},
        {{ 3.0f, -1.5f, 7.0f}, 0.8f},

        {{-4.5f,  0.0f,  9.0f}, 1.2f},
        {{-2.0f,  0.0f,  9.0f}, 1.2f},
        {{ 0.5f,  0.0f,  9.0f}, 1.2f},
        {{ 3.0f,  0.0f,  9.0f}, 1.2f},

        {{-4.0f,  2.5f, 11.0f}, 1.0f},
        {{-1.5f,  2.5f, 11.0f}, 1.0f},
        {{ 1.5f,  2.5f, 11.0f}, 1.0f},
        {{ 4.0f,  2.5f, 11.0f}, 1.0f},

        {{-4.0f, -2.5f, 11.0f}, 1.0f},
        {{-1.5f, -2.5f, 11.0f}, 1.0f},
        {{ 1.5f, -2.5f, 11.0f}, 1.0f}
    };


    // config of image
    const int WIDTH  = 8000;
    const int HEIGHT = 6000;
    const int ITERATIONS = 1000;


    unsigned char* imageDev = nullptr;
    Sphere* spheresDev = nullptr;

    cudaMalloc(&imageDev, WIDTH * HEIGHT * sizeof(unsigned char));
    cudaMalloc(&spheresDev, spheresCount * sizeof(Sphere));

    // copy data
    cudaMemcpy(spheresDev, spheres, spheresCount * sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemset(imageDev, 0, WIDTH * HEIGHT * sizeof(unsigned char));


    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // lanuch kernel
    for (int i = 0; i < ITERATIONS; ++i) {
        raycast<<<grid, block>>>(
            imageDev,
            spheresDev,
            spheresCount,
            WIDTH,
            HEIGHT,
            camera
        );
    }

    unsigned char* image = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("=== CUDA Raycast Benchmark ===\n");
    printf("Resolution : %d x %d\n", WIDTH, HEIGHT);
    printf("Pixels     : %d\n", WIDTH * HEIGHT);
    printf("Spheres    : %d\n", spheresCount);
    printf("Block      : %d x %d\n", block.x, block.y);
    printf("Grid       : %d x %d\n", grid.x, grid.y);
    printf("Iterations : %d\n", ITERATIONS);
    printf("Kernel     : %.3f ms\n", milliseconds / ITERATIONS);
    printf("Throughput : %.2f MPixels/s\n",
           (WIDTH * HEIGHT) / (milliseconds / ITERATIONS) / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(image, imageDev, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // display result
    writeImage(image, WIDTH, HEIGHT);

    cudaFree(imageDev);
    cudaFree(spheresDev);
    free(image);

    return 0;
}
