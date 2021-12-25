// Copyright Brendan Ashworth 2021
#include <Python.h>
/* suppress warnings about deprecated apis */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
/* in case M_PI is not enabled */
#ifndef M_PI
#define M_PI 3.141592653589793238462643383
#endif

#include "simd_math_prims.h"

// exported functions
static PyObject *logd(PyObject *self, PyObject *args);
static PyObject *expd(PyObject *self, PyObject *args);
static PyObject *cosd(PyObject *self, PyObject *args);
static PyObject *sind(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    { "log", logd, METH_VARARGS, "Approximate natural logarithm"},
    { "exp", expd, METH_VARARGS, "Approximate natural exponential"},
    { "cos", cosd, METH_VARARGS, "Approximate cosine"},
    { "sin", sind, METH_VARARGS, "Approximate sine"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "approxmath",
    "Fast approximate math functions: log, exp, sin, cos",
    -1, /* module size, but no gc */
    methods,
    /* we don't use garbage collection things, so
     * these options are null */
    NULL, NULL, NULL, NULL,
};

// initialize the module
PyMODINIT_FUNC PyInit_approxmath(void) {
    PyObject* m = PyModule_Create(&moduledef);
    import_array();
    return m;
}

// dank
double trig_canonical(double theta) {
    while (theta > M_PI)
        theta -= 2 * M_PI;

    while (theta < - M_PI)
        theta += 2 * M_PI;

    return theta;
}

#define build_np_func(np_func_name, c_func_name, normalize_trig) \
    static PyObject *np_func_name(PyObject *self, PyObject *args) { \
        PyArrayObject* in_arr; \
        PyArrayObject* out_arr = NULL; \
        npy_intp size; \
        double* inptr, *outptr; \
 \
        if (!PyArg_ParseTuple(args, "O!|O!", &PyArray_Type, &in_arr, &PyArray_Type, &out_arr)) { \
            return NULL; \
        } \
 \
        size = PyArray_SIZE(in_arr); \
        /* create out_arr if it was not already created */ \
        if (out_arr == NULL) { \
            out_arr = (PyArrayObject *) PyArray_NewLikeArray(in_arr, NPY_KEEPORDER, NULL, 1); \
        } else if (size != PyArray_SIZE(out_arr)) { \
            /* size mismatch */ \
            return NULL; \
        } \
 \
        inptr = (double*) PyArray_DATA(in_arr); \
        outptr = (double*) PyArray_DATA(out_arr); \
        while (size--) { \
            if (normalize_trig) \
                outptr[0] = c_func_name(trig_canonical(*inptr)); \
            else \
                outptr[0] = c_func_name(*inptr); \
            inptr++; \
            outptr++; \
        } \
 \
        return (PyObject *) out_arr; \
    }

build_np_func(logd, logapprox_d, 0);
build_np_func(expd, expapprox_d, 0);
build_np_func(cosd, cosapprox_d, 1);
build_np_func(sind, sinapprox_d, 1);
