#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unordered_map>

// Global variable for the dilation factor
static double DILATION_FACTOR = 1.0;  // Default value

// Seed for the RNG (must be non-zero)
static uint32_t rng_state = 1;  // Default seed

// Track current and peak memory use statistics.
static size_t currentMemoryUsage = 0;
static size_t peakMemoryUsage = 0;

// Track object sizes
static std::unordered_map<void *, size_t> objSizes;

// Fast Xorshift RNG function
static inline uint32_t xorshift32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

// Function to seed the RNG
static void seed_xorshift(uint32_t seed) {
    if (seed != 0) {
        rng_state = seed;
    } else {
        rng_state = 1;  // Avoid zero seed
    }
}


// Function declarations
void custom_free(void *ctx, void *ptr);
void *custom_malloc(void *ctx, size_t size);
void *custom_calloc(void *ctx, size_t nelems, size_t size);
void *custom_realloc(void *ctx, void *ptr, size_t size);

// Set the dilation factor
static PyObject *set_dilation_factor(PyObject *self, PyObject *args) {
    double factor;
    if (!PyArg_ParseTuple(args, "d", &factor)) {
      printf("Warning: couldn't parse dilation factor.\n");
      DILATION_FACTOR = 1.0;
      return NULL;  // Parse error
    }
    DILATION_FACTOR = factor;
    Py_RETURN_NONE;
}

static inline double stochastic_round(double v) {
  double q = v - floor(v);
  double r_unif = (double)xorshift32() / (double)UINT32_MAX;
  if (r_unif <= q) {
    return ceil(v);
  } else {
    return floor(v);
  }
}

// Perform probabilistic dilation
static size_t apply_dilation(size_t size, double dilation_factor) {
  //  return (double) size * dilation_factor; // FIXME perhaps, deterministic for now
  size_t new_size = size * stochastic_round(dilation_factor);
  //  printf("[DEBUG] Applying dilation: dilation factor=%f, old size=%lu, new size=%lu\n", dilation_factor, size, new_size);
  return new_size;
}

// The original allocator, used for actually performing allocation operations
static PyMemAllocatorEx origAlloc;

// The wrapping allocator
static PyMemAllocatorEx alloc = {
  .ctx = NULL,
  .malloc = custom_malloc,
  .calloc = custom_calloc,
  .realloc = custom_realloc,
  .free = custom_free,
};

// Custom malloc implementation
void *custom_malloc(void *ctx, size_t size) {
    size_t dilated_size = apply_dilation(size, DILATION_FACTOR);
    void * ptr = origAlloc.malloc(ctx, size);
    objSizes[ptr] = dilated_size;
    currentMemoryUsage += dilated_size;
    if (currentMemoryUsage > peakMemoryUsage) {
      peakMemoryUsage = currentMemoryUsage;
    }
    return ptr;
}

// Custom calloc implementation
void *custom_calloc(void *ctx, size_t nelem, size_t elsize) {
    size_t size = nelem * elsize;
    size_t dilated_size = apply_dilation(size, DILATION_FACTOR);
    //printf("[DEBUG] Allocating %zu bytes (dilated to %zu bytes with factor %.2f).\n",
    //       size, dilated_size, dilation_factor);
    void * ptr = origAlloc.calloc(ctx, size, 1);
    objSizes[ptr] = dilated_size;
    currentMemoryUsage += dilated_size;
    if (currentMemoryUsage > peakMemoryUsage) {
      peakMemoryUsage = currentMemoryUsage;
    }
    return ptr;
}

// Custom realloc implementation
void *custom_realloc(void *ctx, void *ptr, size_t size) {
    if (ptr == NULL) {
        return custom_malloc(ctx, size);
    }
    if (size == 0) {
        custom_free(ctx, ptr);
        return NULL;
    }
    currentMemoryUsage -= objSizes[ptr];
    void * new_ptr = origAlloc.realloc(ctx, ptr, size);
    currentMemoryUsage += size; // FIXME dilate?
    if (currentMemoryUsage > peakMemoryUsage) {
      peakMemoryUsage = currentMemoryUsage;
    }
    return new_ptr;
}

// Custom free implementation
void custom_free(void *ctx, void *ptr) {
  if (ptr) {
    currentMemoryUsage -= objSizes[ptr];
    origAlloc.free(ctx, ptr);
    objSizes[ptr] = 0;
  }
}

// Set the custom allocator
void set_custom_allocator() {
    PyMem_GetAllocator(PYMEM_DOMAIN_OBJ, &origAlloc);
    PyMem_SetAllocator(PYMEM_DOMAIN_MEM, &alloc);
    PyMem_SetAllocator(PYMEM_DOMAIN_OBJ, &alloc);
}

// Python wrapper for clear_statistics
static PyObject* reset_statistics(PyObject* self, PyObject* args) {
  currentMemoryUsage = 0;
  peakMemoryUsage = 0;
  Py_RETURN_NONE;
}

static PyObject* get_current_allocated(PyObject* self, PyObject* args) {
    return PyLong_FromSize_t(currentMemoryUsage);
}

static PyObject* get_peak_allocated(PyObject* self, PyObject* args) {
    return PyLong_FromSize_t(peakMemoryUsage);
}

// Python methods
static PyMethodDef CustomAllocMethods[] = {
    {"set_dilation_factor", set_dilation_factor, METH_VARARGS, "Set the dilation factor."},
    {"reset_statistics", reset_statistics, METH_NOARGS, "Clear allocation statistics."},
    {"get_current_allocated", get_current_allocated, METH_NOARGS, "Get the current allocated memory size."},
    {"get_peak_allocated", get_peak_allocated, METH_NOARGS, "Get the peak allocated memory size."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef customalloc_module = {
    PyModuleDef_HEAD_INIT,
    "customalloc",
    "Custom memory allocator with dilation",
    -1,
    CustomAllocMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_customalloc(void) {
    seed_xorshift(time(NULL));  // Seed the random number generator
    set_custom_allocator();
    return PyModule_Create(&customalloc_module);
}
