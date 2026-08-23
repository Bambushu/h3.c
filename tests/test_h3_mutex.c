#include "../h3_mutex.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void fail(const char *message) {
    fprintf(stderr, "h3_mutex_test: %s\n", message);
    exit(1);
}

int main(void) {
    char directory[] = "/tmp/h3-mutex-test-XXXXXX";
    if (!mkdtemp(directory)) fail("cannot create temporary directory");
    if (setenv("TMPDIR", directory, 1) != 0)
        fail("cannot set temporary directory");

    h3_mutex first;
    if (!h3_mutex_acquire(&first)) fail("first acquire failed");

    pid_t child = fork();
    if (child < 0) fail("cannot fork waiter");
    if (child == 0) {
        /* Do not inherit the parent's held descriptor into the independent
         * acquire being tested. */
        close(first.fd);
        first.fd = -1;
        h3_mutex second;
        if (!h3_mutex_acquire(&second)) _exit(2);
        /* Deliberately omit release: kernel-owned locks must survive crashes
         * and ordinary abnormal process termination without stale state. */
        _exit(0);
    }

    struct timespec delay = {.tv_sec = 0, .tv_nsec = 100000000L};
    if (nanosleep(&delay, NULL) != 0) fail("waiter observation delay failed");
    int status;
    pid_t result = waitpid(child, &status, WNOHANG);
    if (result != 0) fail("waiter acquired the lock before release");

    h3_mutex_release(&first);
    if (waitpid(child, &status, 0) != child)
        fail("cannot collect waiter");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        fail("waiter did not acquire the lock after release");

    h3_mutex third;
    if (!h3_mutex_acquire(&third))
        fail("lock was not released when waiter exited");
    h3_mutex_release(&third);

    char lock_path[PATH_MAX];
    int length = snprintf(lock_path, sizeof(lock_path), "%s/h3.c.lock",
                          directory);
    if (length <= 0 || (size_t)length >= sizeof(lock_path))
        fail("temporary lock path is too long");
    unlink(lock_path);
    rmdir(directory);
    puts("h3_mutex_test: ok");
    return 0;
}
