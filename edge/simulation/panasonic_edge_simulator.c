#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../exported/panasonic_soc_proxy_selected_rf.h"
#include "../exported/panasonic_soc_proxy_rf_50_trees.h"

typedef double (*predict_function)(const float features[4]);

static void print_usage(const char *program_name)
{
    fprintf(stderr,
            "Usage: %s [--model selected|candidate]\n"
            "Read one CSV row per line: voltage_v,current_a,"
            "battery_temp_c,elapsed_time_s\n"
            "Write proxy predictions as CSV to stdout.\n",
            program_name);
}

static int parse_features(char *line, float features[4])
{
    char *cursor = line;
    for (int index = 0; index < 4; ++index) {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
               *cursor == '\n') {
            ++cursor;
        }
        if (*cursor == '\0') {
            return 0;
        }

        errno = 0;
        char *end = NULL;
        const double value = strtod(cursor, &end);
        if (end == cursor || errno == ERANGE || !isfinite(value) ||
            value > FLT_MAX || value < -FLT_MAX) {
            return 0;
        }
        features[index] = (float)value;
        if (!isfinite((double)features[index])) {
            return 0;
        }
        cursor = end;

        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
               *cursor == '\n') {
            ++cursor;
        }
        if (index < 3) {
            if (*cursor != ',') {
                return 0;
            }
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    predict_function predict = panasonic_soc_proxy_selected_rf_predict;
    const char *model_name = "selected_baseline";

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 1 && argc != 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 3) {
        if (strcmp(argv[1], "--model") != 0) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        if (strcmp(argv[2], "selected") == 0) {
            predict = panasonic_soc_proxy_selected_rf_predict;
            model_name = "selected_baseline";
        } else if (strcmp(argv[2], "candidate") == 0) {
            predict = panasonic_soc_proxy_rf_50_trees_predict;
            model_name = "optimized_candidate_50_trees";
        } else {
            fprintf(stderr, "Unknown model '%s'; use selected or candidate.\n",
                    argv[2]);
            return EXIT_FAILURE;
        }
    }

    char line[256];
    unsigned long sample_index = 0;
    printf("sample_index,model,soc_proxy_percent\n");
    while (fgets(line, sizeof(line), stdin) != NULL) {
        const size_t line_length = strlen(line);
        if (line_length > 0 && line[line_length - 1] != '\n' && !feof(stdin)) {
            fprintf(stderr, "Input row %lu is too long.\n", sample_index);
            return EXIT_FAILURE;
        }

        float features[4];
        if (!parse_features(line, features)) {
            fprintf(stderr,
                    "Invalid input row %lu: expected exactly four finite "
                    "comma-separated numbers (voltage,current,temperature,time).\n",
                    sample_index);
            return EXIT_FAILURE;
        }
        const double prediction = predict(features);
        if (!isfinite(prediction)) {
            fprintf(stderr, "Model returned a non-finite prediction at row %lu.\n",
                    sample_index);
            return EXIT_FAILURE;
        }
        printf("%lu,%s,%.17g\n", sample_index, model_name, prediction);
        ++sample_index;
    }
    if (ferror(stdin)) {
        fprintf(stderr, "Failed while reading input.\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
