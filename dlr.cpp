#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<climits>
#include<iostream>
#include<cfloat>
#include<cstring>
#include<sys/stat.h>

#include "accumulate.h"
#include "allreduce.h"
#include "global_data.h"

using std::cout;
using std::string;

typedef unsigned long int ulong_t;

static inline float soft_threshold(float x, float a)
{
	if (fabs(x) < a)
		return 0.0;
	else if (x > 0.0)
		return x - a;
	else
		return x + a;
}

char g_buffer[1<<20];

int parse_line(FILE *file, ulong_t *feature_id, ulong_t *example_id, int *y, float *weight, float *x, fpos_t *fpos)
{
	if (fpos)
		fgetpos(file, fpos);

	if (fgets(g_buffer, 1<<20, file)) {
		char *pstr = strtok(g_buffer, " ");
		*feature_id = strtoul(pstr, NULL, 10);

		pstr = strtok(NULL, " ");
		*example_id = strtoul(pstr, NULL, 10);

		pstr = strtok(NULL, " ");
		*y = strtol(pstr, NULL, 10);

		pstr = strtok(NULL, " ");
		*weight = atof(pstr);

		pstr = strtok(NULL, " ");
		*x = atof(pstr);

		return 1;
	}

	return 0;
}

int parse_line_cache(FILE *file, ulong_t *feature_id, ulong_t *example_id, int *y, float *weight, float *x, fpos_t *fpos)
{
	if (fpos)
		fgetpos(file, fpos);

	if (!fread(feature_id, sizeof(ulong_t), 1, file))
		return 0;
	if (!fread(example_id, sizeof(ulong_t), 1, file))
		return 0;
	if (!fread(y, sizeof(int), 1, file))
		return 0;
	if (!fread(weight, sizeof(float), 1, file))
		return 0;
	if (!fread(x, sizeof(float), 1, file))
		return 0;

	return 1;
}

void update_betaTx(FILE *file, int count, float delta_beta, float *betaTx)
{
	for (int i = 0; i < count; i++) {
		ulong_t feature_id, example_id;
		int y;
		float weight, x; 

		if (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, NULL))
			betaTx[example_id] += delta_beta * x;
	}
}

double get_qloss(float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y)
{
	double qloss = 0.0;

	for (int i = 0; i < feature_count; i++) 
		qloss += lambda_1 * fabs(beta_new[i]);
	
	for (int i = 0; i < example_count; i++) {
		float example_betaTx = betaTx[i];
		float p = 1.0 / (1.0 + exp(-example_betaTx));
		if (p < 1.0e-6)
			p = 1.0e-6;
		if (p > 0.999999)
			p = 0.999999;

		float w = p * (1 - p);
		int y01 = (all_y[i] + 1) / 2;
		float z = example_betaTx + (y01 - p) / w;

		qloss += 0.5 * w * (z - betaTx[i] - betaTx_delta[i]) * (z - betaTx[i] - betaTx_delta[i]);
	}

	return qloss;
}

float* get_grad(char *dataset_filename, ulong_t example_count, ulong_t feature_count, float lambda_1, float *beta)
{
	float *y_betaTx = (float*)calloc(example_count, sizeof(float));
	float *grad = (float*)calloc(feature_count, sizeof(float));
	ulong_t feature_id, example_id;
	int y;
	float weight, x; 

	FILE *file = fopen(dataset_filename, "r");

	while (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, NULL)) {
		y_betaTx[example_id] += y * beta[feature_id] * x;
	}

	rewind(file);

	while (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, NULL)) {
		grad[feature_id] += - y * x / (1.0 + exp(y_betaTx[example_id]));
	}

	for (int i = 0; i < feature_count; i++)
		grad[i] = soft_threshold(grad[i], lambda_1);

	fclose(file);
	free(y_betaTx);

	return grad;
}

double get_grad_norm(char *dataset_filename, ulong_t example_count, ulong_t feature_count, float lambda_1, float *beta)
{
	double grad_norm = 0.0;
	float *grad = get_grad(dataset_filename, example_count, feature_count, lambda_1, beta);

	for (int i = 0; i <= feature_count; i++)
		grad_norm += grad[i] * grad[i];

	free(grad);
	return grad_norm;
}

double get_loss(float alpha, float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y) {

	double loss = 0.0;

	for (int i = 0; i <= feature_count; i++) 
		loss += lambda_1 * fabs((1 - alpha) * beta[i] + alpha * beta_new[i]);

	for (int i = 0; i <= example_count; i++) {
	
		float logit = -all_y[i] * (betaTx[i] + alpha * betaTx_delta[i]);
		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	return loss;
}

void update_feature(FILE *file, ulong_t count, float sum_w_q_x, float sum_w_x_2, float lambda_1, const fpos_t *feature_start_pos, ulong_t feature_id, float *beta, float *betaTx_delta)
{
	float beta_after_cd = 0.0;
		
	if (sum_w_x_2 != 0.0)
		beta_after_cd = soft_threshold(sum_w_q_x, lambda_1) / sum_w_x_2;
	else
		beta_after_cd = 0.0;

	fpos_t cur_fpos;
	fgetpos(file, &cur_fpos);

	fsetpos(file, feature_start_pos);
	update_betaTx(file, count, beta_after_cd - beta[feature_id], betaTx_delta);

	fsetpos(file, &cur_fpos);

	beta[feature_id] = beta_after_cd;
}

void create_cache(char *dataset_filename, char *cache_filename)
{
	cout << "Creating cache " << cache_filename << "\n";

	FILE *file_dataset = fopen(dataset_filename, "r");
	FILE *file_cache = fopen(cache_filename, "w");

	ulong_t prev_feature_id = 0;
	ulong_t prev_example_id = 0;
	ulong_t line_num = 1;

	while (!feof(file_dataset)) {

		ulong_t feature_id, example_id;
		int y;
		float weight, x; 

		while (parse_line(file_dataset, &feature_id, &example_id, &y, &weight, &x, NULL)) {

			if ((prev_feature_id == feature_id) && (prev_example_id == example_id) && (line_num != 1)) // bad situation
				continue;

			fwrite(&feature_id, sizeof(feature_id), 1, file_cache);
			fwrite(&example_id, sizeof(example_id), 1, file_cache);
			fwrite(&y, sizeof(y), 1, file_cache);
			fwrite(&weight, sizeof(weight), 1, file_cache);
			fwrite(&x, sizeof(x), 1, file_cache);

			prev_feature_id = feature_id;
			prev_example_id = example_id;
			line_num++;
		}
	}

	fclose(file_dataset);
	fclose(file_cache);
}

int file_exists(char * filename)
{
	struct stat buf;
	int ret = stat(filename, &buf );

	return (ret == 0);
}

double get_combined_loss(float alpha[3], float *betaTx, float *betaTx_delta[3], float *beta, float *beta_new[3], ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y) {

	double loss = 0.0;

	for (int i = 0; i <= feature_count; i++) {
		double beta_final = beta[i];

		for (int k = 0; k < 3; k++) {
			beta_final += alpha[k] * (beta_new[k][i] - beta[i]);
		}

		loss += lambda_1 * fabs(beta_final);
	}

	for (int i = 0; i <= example_count; i++) {
		float logit = betaTx[i];
		for (int k = 0; k < 3; k++) 
			logit += alpha[k] * betaTx_delta[k][i];

		logit *= -all_y[i];

		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	return loss;
}

void combine_vectors(string master_location, float *betaTx_delta, float *beta_new, ulong example_count)
{
	char *buffer[2];
	buffer[0] = (char*)calloc(example_count, sizeof(float));
	buffer[1] = (char*)calloc(example_count, sizeof(float));

	get_kids_vectors(master_location, (char*)betaTx_delta, buffer, example_count * sizeof(float), global.unique_id, global.total, global.node);
}

int main(int argc, char **argv)
{
	if (argc < 5) {
		printf("Usage: dlr dataset example_count feature_count lambda_1 [master node total]");
		return 0;
	}

	char *dataset_filename = argv[1];
	ulong_t example_count = strtol(argv[2], NULL, 10);
	ulong_t feature_count = strtol(argv[3], NULL, 10);
	float lambda_1 = atof(argv[4]);
	time_t start_time;

	time(&start_time);

	float *betaTx = (float*)calloc(example_count, sizeof(float));
	float *betaTx_delta = (float*)calloc(example_count, sizeof(float));
	int *all_y = (int*)calloc(example_count, sizeof(float));
	float *beta = (float*)calloc(feature_count, sizeof(float));
	float *beta_new = (float*)calloc(feature_count, sizeof(float));

	bool distributed = (argc == 8);
	string master_location;
	extern global_data global;

	if (distributed) {
		//
		// All Reduce preparation
		//	
		master_location = string(argv[5]);
		global.unique_id = 123456;
		global.node = atoi(argv[6]);
		global.total = atoi(argv[7]);
	}


	int length = strlen(dataset_filename);
	char cache_filename[length + 6];
	sprintf(cache_filename, "%s.cache", dataset_filename);
	if (!file_exists(cache_filename))
		create_cache(dataset_filename, cache_filename);

	//
	// Read all_y
	//
	FILE *file = fopen(cache_filename, "r");
	float tmp;

	while (!feof(file)) {
		ulong_t feature_id, example_id;
		int y;
		float weight, x; 

		while (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, NULL)) {
			all_y[example_id] = y;
		}
	}

	fclose(file);
	
	//
	// Synchronize all_y at nodes (some nodes may not have full list of the examples)
	//
	if (distributed) {	
		float *all_y_float = (float*)calloc(example_count, sizeof(float));

		for (int i = 0; i < example_count; i++)			
			all_y_float[i] = all_y[i];

		accumulate_vector(master_location, all_y_float, example_count);
		printf("network time, sec %f \n", get_comm_time() * 1.0e-3);

		for (int i = 0; i < example_count; i++)			
			all_y[i] = (all_y_float[i] < 0 ? -1 : 1);

		free(all_y_float);
	}


	double prev_newton_loss = get_loss(0.0, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
	printf("loss %f\n", prev_newton_loss);
	double qloss = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
	printf("qloss %f\n", qloss);
	double prev_qloss = qloss;
	ulong_t count = 0;
	int cd_count = 0;
	int feature_idx = 0;

	for (int iter = 1; iter < 100; iter++) {

		FILE *file = fopen(cache_filename, "r");
		fpos_t feature_start_pos;
		ulong_t prev_feature_id = 0;

		double sum_w_x_2 = 0.0;
		double sum_w_q_x = 0.0;

		while (!feof(file)) {

			fpos_t tmp_file_pos;

			ulong_t feature_id, example_id;
			int y;
			float weight, x; 

			while (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, &tmp_file_pos)) {

				if (feature_id != prev_feature_id) {

					if (prev_feature_id != 0) {
						update_feature(file, count, sum_w_q_x, sum_w_x_2, lambda_1, &feature_start_pos, prev_feature_id, beta_new, betaTx_delta);

						/*if (feature_idx % 323456 == 0) {
							double qloss = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
							printf("feature_id = %d qloss %f\n", prev_feature_id, qloss);
							printf("feature_id = %d beta = %f count = %d\n", prev_feature_id, beta_new[prev_feature_id], count);
						}
						feature_idx++;*/
					}

			 		feature_start_pos = tmp_file_pos;

					sum_w_x_2 = 0.0;
					sum_w_q_x = 0.0;
					count = 0;
				}

				float example_betaTx = betaTx[example_id];

				float p = 1.0 / (1.0 + exp(-example_betaTx));

				if (p < 1.0e-6)
					p = 1.0e-6;
				if (p > 0.999999)
					p = 0.999999;

				float w = p * (1 - p);
				int y01 = (y + 1) / 2;
				float z = example_betaTx + (y01 - p) / w;

				float q = z - (example_betaTx + betaTx_delta[example_id] - beta_new[feature_id] * x);
				sum_w_x_2 += w * x * x;
				sum_w_q_x += w * q * x;
			
				count++;
				prev_feature_id = feature_id;
			}
		}

		update_feature(file, count, sum_w_q_x, sum_w_x_2, lambda_1, &feature_start_pos, prev_feature_id, beta_new, betaTx_delta);

		//
		// Accumulate deltas of beta
		//
		if (distributed) {	
			accumulate_vector(master_location, betaTx_delta, example_count);
			accumulate_vector(master_location, beta_new, feature_count);
			printf("network time, sec %f \n", get_comm_time() * 1.0e-3);

			for (int i = 0; i < feature_count; i++)
				beta_new[i] -= (global.total - 1) * beta[i];
		}

		//
		// Linear search
		//
		double min_loss = DBL_MAX;
		float best_alpha = 0.0;

		printf("\n");

		for (int i = 0; i <= 22; i++) {
			double alpha;

			if (i < 11)
				alpha = pow(0.5, i);
			else if (i == 11)
				alpha = 0.0;
			else
				alpha = -pow(0.5, 22 - i);

			double loss = get_loss(alpha, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
			
			printf("alpha %f loss %f \n", alpha, loss);

			if (loss < min_loss) {
				best_alpha = alpha;
				min_loss = loss;
			}
		}

		//
		// Check termination criteria
		//
		float grad_norm = 0.0;
		//float grad_norm = get_grad_norm(dataset_filename, example_count, feature_count, lambda_1, beta_new);

		float termination_eps = 1.0e-3;
		
		double loss_new = get_loss(best_alpha, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
		double relative_loss_diff = ((loss_new - prev_newton_loss) / prev_newton_loss);
		printf("loss %f grad_norm %f relative_loss_diff %f\n ", loss_new, grad_norm / feature_count, relative_loss_diff);
		
		int make_newton;

		if (fabs(relative_loss_diff) < termination_eps) {
			if (cd_count < 3) {
				make_newton = 0;
			}
			else {
				make_newton = 1;
				cd_count = 0;
			}
		}
		else {
			make_newton = 1;
			cd_count = 0;
		}

		cd_count++;

		//
		// Make step
		//
		time_t timer;
		time(&timer);

		cout << "iter " << iter << " CD best_alpha " << best_alpha << " min_loss " << min_loss << " time " << (timer - start_time) << "\n";

		if (make_newton) {
			for (int i = 0; i <= example_count; i++) {
				betaTx[i] += best_alpha * betaTx_delta[i];
				betaTx_delta[i] = 0.0;
			}

			for (int i = 0; i <= feature_count; i++) {
				beta[i] = (1 - best_alpha) * beta[i] + best_alpha * beta_new[i];
				beta_new[i] = beta[i];
			}
			time(&timer);
			cout << "iter " << iter << " NW best_alpha " << best_alpha << " min_loss " << min_loss << " time " << (timer - start_time) << "\n";
		}
	
		fclose(file);

		if (make_newton && (fabs(relative_loss_diff) < termination_eps)) {
			break;
		}

		if (make_newton) {
			prev_newton_loss = loss_new;
		}
		
		cout << "\n";
	}

	//
	// Writing features to file
	//
	FILE *file_rfeatures = fopen("rfeatures", "w");

	fprintf(file_rfeatures, "solver_type L1R_LR\n");
	fprintf(file_rfeatures, "nr_class 2\n");
	fprintf(file_rfeatures, "label 1 -1\n");
	fprintf(file_rfeatures, "nr_feature %ld\n", feature_count);
	fprintf(file_rfeatures, "bias -1\n");
	fprintf(file_rfeatures, "w\n");

	for (int i = 1; i < feature_count; i++) {
		fprintf(file_rfeatures, "%f\n", beta[i]);
	}

	fclose(file_rfeatures);

	free(betaTx);
	free(betaTx_delta);
	free(all_y);
	free(beta);
	free(beta_new);

	return 0;
}
