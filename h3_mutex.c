#include "h3_mutex.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/types.h>
#include <unistd.h>

static int mutex_path(char destination[PATH_MAX]) {
    const char *directory = getenv("TMPDIR");
    if (!directory || !*directory) directory = "/tmp";
    int length = snprintf(destination, PATH_MAX, "%s%s%s",
                          directory,
                          directory[strlen(directory) - 1] == '/' ? "" : "/",
                          "h3.c.lock");
    return length > 0 && length < PATH_MAX;
}

int h3_mutex_acquire(h3_mutex *mutex) {
    if (!mutex) return 0;
    memset(mutex, 0, sizeof(*mutex));
    mutex->fd = -1;
    if (!mutex_path(mutex->path)) {
        fprintf(stderr, "h3: lock path is too long\n");
        return 0;
    }
    fprintf(stderr,
            "h3: attempting to acquire file lock at %s. This may pause "
            "if other h3 instances are running\n", mutex->path);
    fflush(stderr);

    mutex->fd = open(mutex->path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (mutex->fd < 0) {
        fprintf(stderr, "h3: cannot open mutex %s: %s\n", mutex->path,
                strerror(errno));
        return 0;
    }

    int result;
    do {
        result = flock(mutex->fd, LOCK_EX);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
        fprintf(stderr, "h3: cannot acquire mutex %s: %s\n",
                mutex->path, strerror(errno));
        close(mutex->fd);
        mutex->fd = -1;
        return 0;
    }

    fprintf(stderr, "h3: acquired file lock\n");
    fflush(stderr);
    return 1;
}

void h3_mutex_release(h3_mutex *mutex) {
    if (!mutex || mutex->fd < 0) return;
    if (flock(mutex->fd, LOCK_UN) != 0) {
        fprintf(stderr, "h3: cannot release mutex %s: %s\n", mutex->path,
                strerror(errno));
    } else {
        fprintf(stderr, "h3: released file lock\n");
        fflush(stderr);
    }
    close(mutex->fd);
    mutex->fd = -1;
}
