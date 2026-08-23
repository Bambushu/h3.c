/* Process-wide serialization for the h3 command-line executable. */
#ifndef H3_MUTEX_H
#define H3_MUTEX_H

#include <limits.h>

typedef struct {
    int fd;
    char path[PATH_MAX];
} h3_mutex;

/* Acquire the process-wide H3 lock. Returns zero and prints an error on
 * failure. The lock is released automatically by the kernel if the process
 * exits unexpectedly. */
int h3_mutex_acquire(h3_mutex *mutex);

void h3_mutex_release(h3_mutex *mutex);

#endif
