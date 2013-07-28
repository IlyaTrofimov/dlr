#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<climits>
#include<iostream>
#include<map>
#include<cfloat>
#include<cstring>
#include<sys/stat.h>
#include "asa047.h"

#include "accumulate.h"
#include "allreduce.h"
#include "global_data.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

using std::cout;
using std::cerr;
using std::string;
using std::map;

time_t g_start_time;
time_t g_time;
string g_master_location;
map<string, int> g_timers;
map<string, int> g_start_timers;

void print_time()
{
	time_t timer;
	time(&timer);
	printf("Time %ld\n", timer - g_start_time);
}

void start_timer(const char* name)
{
	time_t timer;
	time(&timer);

	string key(name);

	if (g_start_timers.find(key) == g_start_timers.end()) {
		g_start_timers[key] = timer;
	}
	else {
		fprintf(stderr, "Error! Timer %s already started\n", key.c_str());
	}
}

void stop_timer(const char* name)
{
	time_t timer;
	time(&timer);
	
	string key(name);

	if (g_start_timers.find(key) == g_start_timers.end()) {
		fprintf(stderr, "Error! Timer %s not started\n", key.c_str());
		return;
	}
	else {
		if (g_timers.find(key) == g_timers.end()) {
			g_timers[key] = timer - g_start_timers[key];
		}
		else {
			g_timers[key] += timer - g_start_timers[key];
		}

		g_start_timers.erase(key);
	}
}

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

int parse_line(FILE *file, ulong *feature_id, ulong *example_id, int *y, float *weight, float *x, fpos_t *fpos)
{
	if (fpos)
		fgetpos(file, fpos);

	if (fgets(g_buffer, 1<<20, file)) {
		char *pstr = strtok(g_buffer, " \t");
		*feature_id = strtoul(pstr, NULL, 10);

		pstr = strtok(NULL, " \t");
		*example_id = strtoul(pstr, NULL, 10);

		pstr = strtok(NULL, " \t");
		*y = strtol(pstr, NULL, 10);

		pstr = strtok(NULL, " \t");
		*weight = atof(pstr);

		pstr = strtok(NULL, " \t");
		*x = atof(pstr);

		return 1;
	}

	return 0;
}

int parse_line_cache(FILE *file, ulong *feature_id, ulong *example_id, int *y, float *weight, float *x, fpos_t *fpos)
{
	if (fpos)
		fgetpos(file, fpos);

	if (!fread(feature_id, sizeof(ulong), 1, file))
		return 0;
	if (!fread(example_id, sizeof(ulong), 1, file))
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
		ulong feature_id, example_id;
		int y;
		float weight, x; 

		if (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, NULL))
			betaTx[example_id] += delta_beta * x;
	}
}

double get_qloss(float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong example_count, ulong feature_count, float lambda_1, int *all_y)
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

float* get_grad(const char *cache_filename, ulong example_count, ulong feature_count, float lambda_1, float *beta)
{
	float *y_betaTx = (float*)calloc(example_count, sizeof(float));
	float *grad = (float*)calloc(feature_count, sizeof(float));
	ulong feature_id, example_id;
	int y;
	float weight, x; 

	FILE *file = fopen(cache_filename, "r");

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

double get_grad_norm(const char *cache_filename, ulong example_count, ulong feature_count, float lambda_1, float *beta)
{
	double grad_norm = 0.0;
	float *grad = get_grad(cache_filename, example_count, feature_count, lambda_1, beta);

	for (int i = 0; i < feature_count; i++)
		grad_norm += grad[i] * grad[i];

	free(grad);
	return grad_norm;
}

double get_loss(float alpha, float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong example_count, ulong feature_count, float lambda_1, int *all_y) {

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

void update_feature(FILE *file, ulong count, float sum_w_q_x, float sum_w_x_2, float lambda_1, const fpos_t *feature_start_pos, ulong feature_id, float *beta, float *betaTx_delta)
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

void create_cache(const char *dataset_filename, const char *cache_filename, ulong *max_feature_id, ulong *max_example_id, ulong *lines_count)
{
	start_timer("creating cache");
	cout << "Creating cache " << cache_filename << "\n";

	FILE *file_dataset = fopen(dataset_filename, "r");
	FILE *file_cache = fopen(cache_filename, "w");

	ulong prev_feature_id = 0;
	ulong prev_example_id = 0;
	ulong line_num = 0;
	*max_feature_id = 0;
	*max_example_id = 0;

	while (!feof(file_dataset)) {

		ulong feature_id, example_id;
		int y;
		float weight, x; 

		while (parse_line(file_dataset, &feature_id, &example_id, &y, &weight, &x, NULL)) {

			line_num++;

			if ((prev_feature_id == feature_id) && (prev_example_id == example_id) && (line_num != 1)) // bad situation
				continue;

			fwrite(&feature_id, sizeof(feature_id), 1, file_cache);
			fwrite(&example_id, sizeof(example_id), 1, file_cache);
			fwrite(&y, sizeof(y), 1, file_cache);
			fwrite(&weight, sizeof(weight), 1, file_cache);
			fwrite(&x, sizeof(x), 1, file_cache);

			if (feature_id > *max_feature_id)
				*max_feature_id = feature_id;

			if (example_id > *max_example_id)
				*max_example_id = example_id;

			prev_feature_id = feature_id;
			prev_example_id = example_id;
		}
	}

	*lines_count = line_num;
	fclose(file_dataset);
	fclose(file_cache);
	stop_timer("creating cache");
}

int file_exists(const char *filename)
{
	struct stat buf;
	int ret = stat(filename, &buf );

	return (ret == 0);
}


float *g_betaTx;
float *g_betaTx_delta[3];
float *g_beta;
float *g_beta_delta[3];
ulong g_example_count;
ulong g_feature_count;
float g_lambda_1;
int *g_all_y;
int g_n;

double get_combined_loss(double alpha[3]) {

	double loss = 0.0;

	for (int i = 0; i < g_feature_count; i++) {
		double beta_final = g_beta[i];

		for (int k = 0; k < g_n; k++) {
			beta_final += alpha[k] * g_beta_delta[k][i];
		}

		loss += g_lambda_1 * fabs(beta_final);
	}

	for (int i = 0; i < g_example_count; i++) {
		float logit = g_betaTx[i];
		for (int k = 0; k < g_n; k++) 
			logit += alpha[k] * g_betaTx_delta[k][i];

		logit *= -g_all_y[i];

		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	return loss;
}

#define SQUARE(x) ((x) * (x))

void print_full_vectors(float *full_vector, ulong reduce_vector_count, char *buffer[2], int *child_sockets)
{
	char filename[32];
	sprintf(filename, "node%ld_%ld.me\0", global.node, global.total);

	FILE *file = fopen(filename, "w");

	for (int i = 0; i < reduce_vector_count; i++)
		fprintf(file, "%f\n", full_vector[i]);

	fclose(file);

	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			sprintf(filename, "node%ld_%ld.child%d\0", global.node, global.total, k);

			FILE *file = fopen(filename, "w");

			for (int i = 0; i < reduce_vector_count; i++)
				fprintf(file, "%f\n", ((float*)buffer[k])[i]);

			fclose(file);
		}
	}
}

void get_linear_combination(double xmin[], int n, int type)
{
	g_n = n;

	for (int i = 0; i < n; i++) {
		xmin[0] = 0.0;
	}

	if (type == 1) {
		// try greedy
			
		int best_idx = 0;
		double min_loss = DBL_MAX;

		for (int i = 0; i < n; i++) {
			xmin[i] = 1.0;

			double loss = get_combined_loss(xmin);
			printf("%d %f\n", i, loss);

			if (loss < min_loss) {
				min_loss = loss;
				best_idx = i;
			}

			xmin[i] = 0.0;
		}

		xmin[best_idx] = 1.0;
		printf("best: \n");

		for (int i = 0; i < n; i++) {
			printf("x[%d] = %f\n", i, xmin[i]);
		}
	}
	else if (type == 2) { // try Nelder-Mead
		double start[n];
		double ynewlo;
		double step[n];
		int konvge = 5;
		int kcount = 100;
		double reqmin = get_combined_loss(xmin);
		int icount;
		int numres;
		int ifault;

		for (int i = 0; i < n; i++) {
			start[i] = 1.0;
			step[i] = 0.5;
			xmin[0] = 0.0;
		}


		nelmin(get_combined_loss, n, start, xmin, &ynewlo, reqmin, step, konvge, kcount, &icount, &numres, &ifault);

		for (int i = 0; i < n; i++) {
			if (fabs(xmin[i]) < 1.0e-2)
				xmin[i] = 0.0;
			if (fabs(xmin[i] - 1.0) < 1.0e-2)
				xmin[i] = 1.0;
		}

		for (int i = 0; i < n; i++) {
			printf("x[%d] = %f icount = %d numres = %d ifault = %d\n", i, xmin[i], icount, numres, ifault);
		}
	}
	else { // discrete opt

		int best_id = 0;
		double min_loss = DBL_MAX;

		printf("\n");

		for (int i = 1; i < (1 << n); i++) {
			for (int j = 0; j < n; j++) {
				xmin[j] = ((i >> j) % 2 == 1);
				printf("x[%d] = %.1f ", j, xmin[j]);
			}

			double loss = get_combined_loss(xmin);
			printf("loss = %f\n", loss);

			if (loss < min_loss) {
				min_loss = loss;
				best_id = i;
			}
		}

		printf("best: \n");

		for (int j = 0; j < n; j++) {
			xmin[j] = ((best_id >> j) % 2 == 1);
				printf("x[%d] = %.1f ", j, xmin[j]);
		}

		printf("\n");
	}
}

ulong max_allreduce(ulong value)
{
	char *buffer[2];
	int *child_sockets;
	buffer[0] = (char*)calloc(1, sizeof(ulong));
	buffer[1] = (char*)calloc(1, sizeof(ulong));

	get_kids_vectors(g_master_location, buffer, sizeof(ulong), global.unique_id, global.total, global.node, &child_sockets);

	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			ulong child_value = *((ulong*)buffer[0]);
			if (child_value > value)
				value = child_value;
		}
	}

	send_to_parent((char*)&value, sizeof(ulong));
	broadcast_buffer((char*)&value, sizeof(ulong));

	return value;
}

void min_allreduce(double x[], int n )
{
	int size = n * sizeof(double);

	char *buffer[2];
	int *child_sockets;
	buffer[0] = (char*)calloc(1, size);
	buffer[1] = (char*)calloc(1, size);

	get_kids_vectors(g_master_location, buffer, size, global.unique_id, global.total, global.node, &child_sockets);

	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			int is_less = 1;

			for (int i = 0; i < n; i++) {
				if (((double*)buffer[k])[i] > x[i]) {
					is_less = 0;
					break;
				}
			}

			if (is_less)
				memcpy(x, buffer[k], size);
		}
	}

	send_to_parent((char*)x, size);
	broadcast_buffer((char*)x, size);
}

void get_best_alpha(float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong example_count, ulong feature_count, float lambda_1, int *all_y, float *best_alpha, double *min_loss)
{
	*min_loss = DBL_MAX;
	*best_alpha = 0.0;

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

		if (loss < *min_loss) {
			*best_alpha = alpha;
			*min_loss = loss;
		}
	}
}

void back_search(double zero_alpha_loss, float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong example_count, ulong feature_count, float lambda_1, int *all_y, 
			float *best_alpha, double *min_loss)
{
	for (int i = 0; i <= 22; i++) {
		double alpha;

		if (i < 11) {
			alpha = pow(0.5, i);
		}
		else if (i == 11) {
			//alpha = 0.0;    is already in loss
			continue;
		}
		else {
			alpha = -pow(0.5, i - 12);
		}

		double loss = get_loss(alpha, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
		
		printf("alpha %f loss %f \n", alpha, loss);

		if (loss < zero_alpha_loss) {
			*best_alpha = alpha;
			*min_loss = loss;
			return;
		}
	}

	*best_alpha = 0.0;
	*min_loss = zero_alpha_loss;
}
void sync_nodes()
{	
	start_timer("sync nodes");
	float tmp_vector[1] = {0.0};
	accumulate_vector(g_master_location, tmp_vector, 1);
	stop_timer("sync nodes");
}

void get_final_coeffs(double xmin[], float *coeffs)
{
	char *buffer_coeffs[2];
	int *child_sockets;
	buffer_coeffs[0] = (char*)calloc(global.total, sizeof(float));
	buffer_coeffs[1] = (char*)calloc(global.total, sizeof(float));

	get_kids_vectors(g_master_location, buffer_coeffs, global.total * sizeof(float), global.unique_id, global.total, global.node, &child_sockets);

	int k = 1;
	if (child_sockets[0] != -1) {
		for (int i = 0; i < global.total; i++)
			coeffs[i] +=  xmin[k] * ((float*)buffer_coeffs[0])[i];
		k++;
	}

	if (child_sockets[1] != -1) {
		for (int i = 0; i < global.total; i++)
			coeffs[i] +=  xmin[k] * ((float*)buffer_coeffs[1])[i];
	}

	send_to_parent((char*)coeffs, global.total * sizeof(float));
	broadcast_buffer((char*)coeffs, global.total * sizeof(float));

	free(buffer_coeffs[0]);
	free(buffer_coeffs[1]);
}

void combine_vectors(float *betaTx_delta, float *beta, float *beta_new, ulong example_count, ulong feature_count, double loss, int type, float coeffs[])
{
	if (type == 0) { // add all deltas

		start_timer("combine vectors - data transfer");
		accumulate_vector(g_master_location, betaTx_delta, example_count);
		accumulate_vector(g_master_location, beta_new, feature_count);
		stop_timer("combine vectors - data transfer");
	
		for (int i = 0; i < feature_count; i++)
			beta_new[i] -= (global.total - 1) * beta[i];

		for (int i = 0; i < global.total; i++)
			coeffs[i] = 1.0;

		return;
	}
	else if (type == 1) {  // select best

		float best_alpha;
		double min_loss;
		memset(coeffs, 0, global.total * sizeof(float));
		
		start_timer("combine vectors - back search");
		back_search(loss, g_betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, g_lambda_1, g_all_y, &best_alpha, &min_loss);
		stop_timer("combine vectors - back search");

		double x[2] = {min_loss, global.node};
		min_allreduce(x, 2);

		if (!(x[0] == min_loss && (int)x[1] == global.node)) {
			memset(betaTx_delta, 0, example_count * sizeof(double));
			memset(beta_new, 0, feature_count * sizeof(double));
			coeffs[global.node] = 1.0;
		}

		start_timer("combine vectors - data transfer");
		accumulate_vector(g_master_location, betaTx_delta, example_count);
		accumulate_vector(g_master_location, beta_new, feature_count);
		stop_timer("combine vectors - data transfer");
		return;
	}

	//
	// Read kids' vectors
	//
	ulong reduce_vector_count = example_count + feature_count;
	char *buffer[2];
	int *child_sockets;
	buffer[0] = (char*)calloc(reduce_vector_count, sizeof(float));
	buffer[1] = (char*)calloc(reduce_vector_count, sizeof(float));

	start_timer("combine vectors - data transfer");
	get_kids_vectors(g_master_location, buffer, reduce_vector_count * sizeof(float), global.unique_id, global.total, global.node, &child_sockets);
	stop_timer("combine vectors - data transfer");

	//
	// Prepare local vector
	//
	start_timer("combine vectors - local opt");
	float *full_vector = (float*)calloc(reduce_vector_count, sizeof(float));

	memcpy(full_vector, betaTx_delta, example_count * sizeof(float));

	for (int i = 0; i < feature_count; i++)
		full_vector[example_count + i] = beta_new[i] - beta[i];

	//print_full_vectors(full_vector, reduce_vector_count, buffer, child_sockets);

	g_betaTx_delta[0] = full_vector;
	g_beta_delta[0] = full_vector + example_count;

	int n = 1;
	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			g_betaTx_delta[n] = (float*)(buffer[k]);
			g_beta_delta[n] = ((float*)(buffer[k]) + example_count);
			n++;
		}
	}

	double xmin[n];

	if (n > 1) {
		get_linear_combination(xmin, n, type);

		//
		// Calculate final linear combination
		//
		for (int i = 0; i < reduce_vector_count; i++) {
			full_vector[i] *= xmin[0];

			int k = 1;
			if (child_sockets[0] != -1) {
				full_vector[i] += xmin[k] * ((float*)buffer[0])[i];
				k++;
			}
			
			if (child_sockets[1] != -1) {
				full_vector[i] += xmin[k] * ((float*)buffer[1])[i];
			}
		}
	}
	stop_timer("combine vectors - local opt");

	start_timer("combine vectors - data transfer");
	send_to_parent((char*)full_vector, reduce_vector_count * sizeof(float));
	broadcast_buffer((char*)full_vector, reduce_vector_count * sizeof(float));
	stop_timer("combine vectors - data transfer");

	memcpy(betaTx_delta, full_vector, example_count * sizeof(float));

	for (int i = 0; i < feature_count; i++)
		beta_new[i] = beta[i] + full_vector[example_count + i];

	free(full_vector);
	free(buffer[0]);
	free(buffer[1]);

	//
	// Calculating final linear coeffs
	//
	memset(coeffs, 0, sizeof(float) * global.total);
	coeffs[global.node] = (n > 1 ? xmin[0] : 1.0);
	get_final_coeffs(xmin, coeffs);
}



void save_beta(const char* filename, float *beta, ulong feature_count)
{
	FILE *file_rfeatures = fopen(filename, "w");

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

}

void print_time_summary()
{
	printf("Time summary:\n");
	int time_total = 0;
	for (map<string, int>::iterator it = g_timers.begin(); it != g_timers.end(); ++it) {
		time_total += it->second;
	}
	
	for (map<string, int>::iterator it = g_timers.begin(); it != g_timers.end(); ++it) {
		printf("%35s: %5ds  %6.1f%%\n", it->first.c_str(), it->second, (float)it->second / time_total * 100);
	}
}

int main(int argc, char **argv)
{
	time(&g_start_time);

	po::options_description general_desc("General options");
	general_desc.add_options()
        	("help,h", "produce help message")
        	("dataset,d", po::value<string>(), "training set in the inverted index form")
        	("final-regressor,f", po::value<string>(), "final weights")
		("lambda-1", po::value<float>()->default_value(1.0), "L1 regularization")
		("combine-type,c", po::value<int>()->default_value(1), "type of deltas combination during AllReduce: \n0 - sum, 1 - greedy, 2 - Nelder-Mead, 3 - discrete opt.")
		("termination", po::value<float>()->default_value(1.0e-4), "termination criteria")
		("iterations", po::value<int>()->default_value(100), "maximum number of iterations")
        	;

	po::options_description cluster_desc("Cluster options");
	cluster_desc.add_options()
        	("server", po::value<string>(), "master server")
        	("unique-id", po::value<int>()->default_value(1), "unique id of the learning transaction")
        	("node", po::value<int>(), "node index")
        	("total", po::value<int>(), "total number of nodes")
        	;

	po::options_description all_desc("Allowed options");
	all_desc.add(general_desc).add(cluster_desc);

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, all_desc), vm);
	po::notify(vm);    

	if (vm.count("help")) {
		cout << general_desc << "\n" << cluster_desc << "\n";
		return 0;
	}

	string dataset_filename = vm["dataset"].as<string>();
	float lambda_1 = vm["lambda-1"].as<float>();
	int combine_type = vm["combine-type"].as<int>();
	float termination_eps = vm["termination"].as<float>();
	int iterations_max = vm["iterations"].as<int>();

	bool distributed = !vm["server"].empty();
	extern global_data global;

	if (distributed) {
		g_master_location = vm["server"].as<string>();
		global.unique_id = vm["unique-id"].as<int>();
		global.node = vm["node"].as<int>();
		global.total = vm["total"].as<int>();

		termination_eps /= global.total;
	}

	//
	// Create cache file
	//
	string cache_filename;

	if (dataset_filename == string("/dev/stdin")) {
		cache_filename = "stdin.cache";
	}
	else {
		cache_filename = dataset_filename + ".cache";
	}

	ulong max_feature_id = 0;
	ulong max_example_id = 0;
	ulong lines_count = 0;

	if (!file_exists(cache_filename.c_str())) {
		create_cache(dataset_filename.c_str(), cache_filename.c_str(), &max_feature_id, &max_example_id, &lines_count);
	}
	else {
		start_timer("initial dataset processing");
		FILE *file = fopen(cache_filename.c_str(), "r");

		while (!feof(file)) {
			ulong feature_id, example_id;
			int y;
			float weight, x; 

			while (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, NULL)) {
				if (feature_id > max_feature_id)
					max_feature_id = feature_id;
				if (example_id > max_example_id)
					max_example_id = example_id;
			}
		}

		fclose(file);
		stop_timer("initial dataset processing");
	}

	if (distributed) {
		start_timer("sync dataset info");
		max_feature_id = max_allreduce(max_feature_id);
		max_example_id = max_allreduce(max_example_id);
		stop_timer("sync dataset info");
	}

	ulong feature_count = max_feature_id + 1;
	ulong example_count = max_example_id + 1;

	printf("lines count   = %ld\n", lines_count);
	printf("feature count = %ld\n", feature_count);
	printf("example count = %ld\n", example_count);
	printf("final regressor = %s\n", vm["final-regressor"].as<string>().c_str());
	printf("combination type = %d\n", combine_type);
	printf("iterations_max = %d\n", iterations_max);

	if (distributed) {
		printf("server = %s\n", g_master_location.c_str());
		printf("node = %ld\n", global.node);
		printf("total = %ld\n", global.total);
		printf("unique id = %ld\n", global.unique_id);
	}

	printf("\n");
	printf("\n");

	//
	// Allocating important vectors
	//
	float *betaTx = (float*)calloc(example_count, sizeof(float));
	float *betaTx_delta = (float*)calloc(example_count, sizeof(float));
	int *all_y = (int*)calloc(example_count, sizeof(float));
	float *beta = (float*)calloc(feature_count, sizeof(float));
	float *beta_new = (float*)calloc(feature_count, sizeof(float));
	float *grad = (float*)calloc(feature_count, sizeof(float));
	//fpos_t *beta_pos = (fpos_t*)calloc(feature_count, sizeof(float));
	//char *active = (char*)calloc(feature_count, 1);

	//memset(active, 1, feature_count);

	//
	// Read all_y
	//
	start_timer("initial dataset processing");
	FILE *file = fopen(cache_filename.c_str(), "r");
	ulong line_idx = 0;
	ulong prev_feature_id = 0;

	while (!feof(file)) {
		ulong feature_id, example_id;
		int y;
		float weight, x; 
		fpos_t fpos;

		while (parse_line_cache(file, &feature_id, &example_id, &y, &weight, &x, &fpos)) {
			all_y[example_id] = y;

			/*if ((feature_id != prev_feature_id) || (line_idx == 0)) {
				beta_pos[feature_id] = fpos;
				prev_feature_id = feature_id;
			}*/	

			line_idx++;
		}
	}

	fclose(file);
	stop_timer("initial dataset processing");
	sync_nodes();
	
	//
	// Synchronize all_y at nodes (some nodes may not have full list of the examples)
	//
	if (distributed) {	
		start_timer("sync all_y");
		float *all_y_float = (float*)calloc(example_count, sizeof(float));

		for (int i = 0; i < example_count; i++)			
			all_y_float[i] = all_y[i];

		accumulate_vector(g_master_location, all_y_float, example_count);
		printf("network time, sec %f \n", get_comm_time() * 1.0e-3);

		for (int i = 0; i < example_count; i++)			
			all_y[i] = (all_y_float[i] < 0 ? -1 : 1);

		free(all_y_float);
		stop_timer("sync all_y");
	}

	double prev_loss = get_loss(0.0, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
	printf("loss %f\n\n", prev_loss);
	//double qloss = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
	//printf("qloss %f\n", qloss);
	ulong count = 0;
	int cd_count = 0;
	int feature_idx = 0;
	file = fopen(cache_filename.c_str(), "r");
	int active_features_iteration = 0;

	for (int iter = 1; iter <= iterations_max; iter++) {
		
		start_timer("iterations");
		printf("Iteration %d\n", iter);
		printf("--------------\n");
		fpos_t feature_start_pos;
		ulong prev_feature_id = 0;

		double sum_w_x_2 = 0.0;
		double sum_w_q_x = 0.0;

		while (!feof(file)) {

			fpos_t tmp_file_pos;
			memset(grad, 0, feature_count * sizeof(float));

			ulong feature_id, example_id;
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

				grad[feature_id] += - y * x / (1.0 + exp(y * example_betaTx));
			
				count++;
				prev_feature_id = feature_id;
			}
		}

		update_feature(file, count, sum_w_q_x, sum_w_x_2, lambda_1, &feature_start_pos, prev_feature_id, beta_new, betaTx_delta);
		stop_timer("iterations");

		sync_nodes();

		double min_loss;
		float best_alpha;
	
		//printf("Testing local delta:\n");
		//start_timer("debug, local lin.search");
		//get_best_alpha(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y, &best_alpha, &min_loss);
		//printf("best_alpha = %f min_loss = %f\n", best_alpha, min_loss);
		//stop_timer("debug, local lin.search");

		//
		// Accumulate deltas of beta
		//
		if (distributed) {
			start_timer("deltas combination");
			g_betaTx = betaTx;
			g_beta = beta;
			g_example_count = example_count;
			g_feature_count = feature_count;
			g_lambda_1 = lambda_1;
			g_all_y = all_y;

			float coeffs[global.total];
			combine_vectors(betaTx_delta, beta, beta_new, example_count, feature_count, prev_loss, combine_type, coeffs);

			printf("\n");
			float c_abs_norm = 0.0;
			for (int i = 0; i < global.total; i++) {
				printf("c[%d] = %.2f ", i, coeffs[i]);
				c_abs_norm += fabs(coeffs[i]);
			}
		
			printf("\n");
			printf("||c||_1 = %f\n", c_abs_norm);
			printf("\n");
		
			printf("network time, sec %f \n", get_comm_time() * 1.0e-3);
			stop_timer("deltas combination");
		}

		start_timer("calc gradient");
		double grad_norm = 0.0;
		for (int i = 0; i < feature_count; i++) {
			grad[i] = soft_threshold(grad[i], lambda_1);
			grad_norm += SQUARE(grad[i]);
		}
		stop_timer("calc gradient");

		//printf("gradient norm  = %f\n", grad_norm);

		//double grad_norm2 = get_grad_norm(cache_filename.c_str(), example_count, feature_count, lambda_1, beta);
		//printf("gradient norm2 = %f\n", grad_norm2);

		//
		// Linear search
		//
		start_timer("linear search");
		printf("Testing combined delta:\n");
		back_search(prev_loss, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y, &best_alpha, &min_loss);
		printf("\n");
		stop_timer("linear search");

		//
		// Check termination criteria
		//
		double loss_new = get_loss(best_alpha, betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
		double relative_loss_diff = ((loss_new - prev_loss) / prev_loss);
		printf("loss %e ||grad||_2 %e relative_loss_diff %e\n", loss_new, grad_norm, relative_loss_diff);
		
		int make_newton;
		int cd_max = 1;
		cd_count++;
 
		if (fabs(relative_loss_diff) < termination_eps) {
			if (cd_count < cd_max) {
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

		//
		// Make step
		//
		cout << "iter " << iter << " CD best_alpha " << best_alpha << " min_loss " << min_loss << "\n";

		if (make_newton) {
			for (int i = 0; i <= example_count; i++) {
				betaTx[i] += best_alpha * betaTx_delta[i];
				betaTx_delta[i] = 0.0;
			}

			for (int i = 0; i <= feature_count; i++) {
				beta[i] = (1 - best_alpha) * beta[i] + best_alpha * beta_new[i];
				beta_new[i] = beta[i];
			}
			cout << "iter " << iter << " NW best_alpha " << best_alpha << " min_loss " << min_loss << "\n";
		}

		print_time();
	
		if (make_newton && (fabs(relative_loss_diff) < termination_eps)) {
			double grad_norm2 = get_grad_norm(cache_filename.c_str(), example_count, feature_count, lambda_1, beta_new);
			printf("||grad||_2 = %e\n", grad_norm2);
			break;
		}

		if (make_newton) {
			prev_loss = loss_new;
		}

		rewind(file);
		
		cout << "\n";
	}

	fclose(file);

	printf("\n");	
	print_time_summary();

	save_beta(vm["final-regressor"].as<string>().c_str(), beta, feature_count);

	free(betaTx);
	free(betaTx_delta);
	free(all_y);
	free(beta);
	free(beta_new);
	free(grad);
	//free(beta_pos);
	//free(active);

	return 0;
}
