/* PRIMA BOBYQA subprocess bridge. All coordinates are in [0, 1].
 * stdout: EVAL x...; stdin: objective value; stdout: RESULT rc x... f nfev.
 */
#include "prima/prima.h"
#include <stdio.h>
#include <stdlib.h>

static int dimension;
static void objective(const double x[], double *f, const void *data) {
    (void)data;
    fputs("EVAL", stdout);
    for (int i = 0; i < dimension; i++) printf(" %.17g", x[i]);
    fputc('\n', stdout);
    fflush(stdout);
    if (scanf("%lf", f) != 1) exit(3);
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    dimension = atoi(argv[1]);
    int maxfun = atoi(argv[2]);
    if (dimension < 1 || argc != dimension + 3) return 2;
    double *x0 = calloc(dimension, sizeof(double));
    double *xl = calloc(dimension, sizeof(double));
    double *xu = calloc(dimension, sizeof(double));
    if (!x0 || !xl || !xu) return 2;
    for (int i = 0; i < dimension; i++) {
        x0[i] = atof(argv[i + 3]);
        xu[i] = 1.0;
    }
    prima_problem_t problem;
    prima_options_t options;
    prima_result_t result;
    prima_init_problem(&problem, dimension);
    prima_init_options(&options);
    problem.x0 = x0;
    problem.xl = xl;
    problem.xu = xu;
    problem.calfun = objective;
    options.rhobeg = 0.1;
    options.rhoend = 1e-6;
    options.maxfun = maxfun;
    int rc = prima_minimize(PRIMA_BOBYQA, problem, options, &result);
    if (!result.x) {
        fprintf(stderr, "PRIMA BOBYQA returned no solution (status %d)\n", rc);
        return 4;
    }
    printf("RESULT %d", rc);
    for (int i = 0; i < dimension; i++) printf(" %.17g", result.x[i]);
    printf(" %.17g %d\n", result.f, result.nf);
    fflush(stdout);
    prima_free_result(&result);
    free(x0);
    free(xl);
    free(xu);
    return 0;
}
