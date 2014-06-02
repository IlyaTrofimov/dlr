#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<climits>
#include<iostream>
#include<sstream>
#include<map>
#include<cfloat>
#include<cstring>
#include<vector>
#include<algorithm>
#include<sys/stat.h>
#include<algorithm>
#include "asa047.h"

#include "accumulate.h"
#include "allreduce.h"
#include "global_data.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

typedef uint32_t example_t;
typedef uint32_t feature_t;

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::map;
using std::vector;
using std::random_shuffle;
using std::fill;
using std::to_string;

#define SGN(x) (((x) > 0)  - ((x) < 0))
#define SQUARE(x) ((x) * (x))
#define LIMIT(x, min, max) ( (x) < min ? min : ( (x) > max ? max : (x))) 

#define NU 1.e-6
#define SIGMA 0.01
#define MAX_betaTx 100.0
#define P_MIN 1.0e-6
#define P_MAX (1.0 - 1.0e-6)
#define BACK_SEARCH_ARMIJO_MAX 10

//
// Global variables 
//
//float *g_betaTx;
float *g_exp_betaTx;

float *g_betaTx_delta;
float *g_beta;
float *g_beta_new;
float *g_beta_delta[3];
feature_t g_example_count;
example_t g_feature_count;
float g_lambda_1;
int *g_all_y;

float *g_betaTx_delta_c[3];
int g_n;

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

size_t allocated = 0;
map<void*, size_t> pointers;

void* safe_calloc(size_t num, size_t size)
{
	void *p = (float*)calloc(num, size);
	if (!p) {
		cerr << "Cannot allocate memory!" << endl;
		exit(EXIT_FAILURE);
	}

	pointers[p] = num * size;
	allocated += num * size;

	cout << "Allocated " << num * size / 1024 / 1024 << ", total " << allocated / 1024 / 1024 << " Mb" << endl;

	return p;
}

void safe_free(void *p)
{
	if (!p) {
		cerr << "Cannot free bad pointer!" << endl;
		exit(EXIT_FAILURE);
	}

	allocated -= pointers[p];
	pointers.erase(p);
	free(p);	
	cout << "Allocated total " << allocated / 1024 / 1024 << " Mb" << endl;
}

void print_coeffs(const char *name, float *coeffs, int count) 
{
	printf("%s:\n", name);

	for (int i = 0; i < count; i++) {
		printf("%5.2f ", coeffs[i]);
	}
	printf("\n");
}

void print_coeffs_style(const char *name, const char *style, float *coeffs, int count) 
{
	printf("%s:\n", name);

	for (int i = 0; i < count; i++) {
		printf(style, coeffs[i]);
	}
	printf("\n");
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

void sync_nodes()
{	
	start_timer("sync nodes");
	float tmp_vector[1] = {0.0};
	accumulate_vector(g_master_location, tmp_vector, 1);
	stop_timer("sync nodes");
}

string to_string(int n, const char *format) 
{
	char buffer[256];
	sprintf(buffer, format, n);
	return string(buffer);
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

int parse_line(FILE *file, feature_t *feature_id, example_t *example_id, int *y, float *weight, float *x, fpos_t *fpos)
{
	if (fpos)
		fgetpos(file, fpos);

	if (fgets(g_buffer, sizeof(g_buffer), file)) {
		char *pstr = strtok(g_buffer, " \t");
		*feature_id = strtoul(pstr, NULL, 10);

		pstr = strtok(NULL, " \t");
		*example_id = strtoul(pstr, NULL, 10);

		//pstr = strtok(NULL, " \t");
		//*y = strtol(pstr, NULL, 10);

		//pstr = strtok(NULL, " \t");
		//*weight = atof(pstr);

		pstr = strtok(NULL, " \t");
		*x = atof(pstr);

		return 1;
	}

	return 0;
}

class Cache
{
public:
	Cache() {};
	void Create(const char *dataset_filename, const char *cache_filename, feature_t *max_feature_id, ulong *lines_count, example_t *unique_features);
	int ReadVariable(feature_t *feature_id);
	int ReadLine(example_t *example_id, float *x, int *y, float *weight);
	void MoveToStartVariable();
	void Rewind();
	int* GetY();
	feature_t GetFeatureId(int idx);
	feature_t GetFeatureIndex(feature_t feature_count);
	void InitY(example_t example_count, feature_t feature_count, bool distributed, const po::variables_map& vm);
	void ReadY(const char *labels_filename, example_t *max_example_id);
	void InitCache(feature_t feature_count);
	void MoveToVar(feature_t feature_id);
	feature_t GetCacheFeatureCount();
	~Cache();

private:
	FILE *_file;
	feature_t _feature_id;
	bool _is_new_feature;
	fpos_t _start_feature_pos;
	int *_all_y;

	vector<float> _y;
	vector<float> _weight;	
	vector<fpos_t> _start_var_pos;
	vector<feature_t> _features;
	vector<feature_t> _features_index;
	feature_t _cache_feature_count;
};

Cache g_cache;

Cache::~Cache()
{
	fclose(_file);
	safe_free(_all_y);
}

int* Cache::GetY()
{
	return _all_y;
}

feature_t Cache::GetCacheFeatureCount()
{
	return _cache_feature_count;
}

feature_t Cache::GetFeatureId(int idx) 
{
	return _features[idx];
}

feature_t Cache::GetFeatureIndex(feature_t feature_id) 
{
	return _features_index[feature_id];
}

void Cache::InitCache(feature_t feature_count)
{
	feature_t feature_id;
	feature_t i = 0;

	_start_var_pos.resize(feature_count);
	_features.resize(feature_count);
	_features_index.resize(feature_count);

	Rewind();
	
	while(ReadVariable(&feature_id)) {
		_features[i] = feature_id;
		_features_index[feature_id] = i;
		_start_var_pos[feature_id] = _start_feature_pos;
		i++;

		example_t example_id;
		int y;
		float weight, x;

		while(ReadLine(&example_id, &x, &y, &weight)) {	}
	}

	_cache_feature_count = i;
}

void Cache::InitY(example_t example_count, feature_t feature_count, bool distributed, const po::variables_map& vm)
{
	Rewind();
	_all_y = (int*)safe_calloc(example_count, sizeof(int));

	//
	// Synchronize all_y at nodes (some nodes may not have full list of the examples)
	//
	if (distributed) {
		if (vm["sync"].as<int>()) sync_nodes();

		start_timer("sync all_y");
		float *all_y_float = (float*)safe_calloc(example_count, sizeof(float));

		for (int i = 0; i < (int)_y.size(); i++)			
			all_y_float[i] = _y[i];

		accumulate_vector(g_master_location, all_y_float, example_count);
		printf("network time, sec %f \n", get_comm_time() * 1.0e-3);

		for (int i = 0; i < (int)example_count; i++)			
			_all_y[i] = (all_y_float[i] < 0 ? -1 : 1);

		safe_free(all_y_float);
		stop_timer("sync all_y");
	}
	else {
		for (int i = 0; i < example_count; i++)			
			_all_y[i] = _y[i];
	}

	// clear contents and free memory
	vector<float>().swap(_y); 
	vector<float>().swap(_weight);
}

void Cache::ReadY(const char *labels_filename, example_t *max_example_id)
{
	start_timer("reading labels");
	cout << "Reading labels " << labels_filename << "\n";

	FILE *file = fopen(labels_filename, "r");

	*max_example_id = 0;
	_y.resize(0);
//	_weight.resize(0);

	while (!feof(file)) {

		example_t example_id;
		int y;
		float weight; 
		if (fgets(g_buffer, sizeof(g_buffer), file)) {
			char *pstr = strtok(g_buffer, " \t");
			example_id = strtoul(pstr, NULL, 10);

			pstr = strtok(NULL, " \t");
			y = strtol(pstr, NULL, 10);

			pstr = strtok(NULL, " \t");
			weight = atof(pstr);

			if (example_id > *max_example_id)
				*max_example_id = example_id;

			if (example_id + 1 > _y.size()) {
				_y.resize(example_id + 1);
				_weight.resize(example_id + 1);
			}
			
			_y[example_id] = y;
//			_weight[example_id] = weight;
		}
	}

	fclose(file);
	stop_timer("reading labels");

	example_t example_count = *max_example_id + 1;

	_all_y = (int*)safe_calloc(example_count, sizeof(int));

	for (int i = 0; i < (int)example_count; i++)			
		_all_y[i] = _y[i];

	// clear contents and free memory
	vector<float>().swap(_y); 
}

void Cache::Create(const char *dataset_filename, const char *cache_filename, feature_t *max_feature_id, ulong *lines_count, feature_t *unique_features)
{
	start_timer("creating cache");
	cout << "Creating cache " << cache_filename << "\n";

	FILE *file_dataset = fopen(dataset_filename, "r");
	FILE *file_cache = fopen(cache_filename, "w");

	feature_t prev_feature_id = 0;
	example_t prev_example_id = 0;
	ulong line_num = 0;
	*max_feature_id = 0;
	*unique_features = 0;

	_y.resize(0);
	_weight.resize(0);

	while (!feof(file_dataset)) {

		feature_t feature_id;
		example_t example_id;
		int y;
		float weight, x; 

		while (parse_line(file_dataset, &feature_id, &example_id, &y, &weight, &x, NULL)) {

			line_num++;

			if ((prev_feature_id == feature_id) && (prev_example_id == example_id) && (line_num != 1)) // bad situation
				continue;

			if (prev_feature_id != feature_id) {
				(*unique_features)++;
				
				example_t zero_example = 0;	
				float zero_x = 0.0;

				if (line_num != 1) {
					fwrite(&zero_example, sizeof(zero_example), 1, file_cache);
					fwrite(&zero_x, sizeof(zero_x), 1, file_cache);
				}

				fwrite(&feature_id, sizeof(feature_id), 1, file_cache);
			}

			fwrite(&example_id, sizeof(example_id), 1, file_cache);
			fwrite(&x, sizeof(x), 1, file_cache);

			if (feature_id > *max_feature_id)
				*max_feature_id = feature_id;

			prev_feature_id = feature_id;
			prev_example_id = example_id;
		}
	}

	*lines_count = line_num;
	fclose(file_dataset);
	fclose(file_cache);
	stop_timer("creating cache");

	_file = fopen(cache_filename, "r");
	_is_new_feature = true;
	fgetpos(_file, &_start_feature_pos);
}

void Cache::MoveToStartVariable()
{
	fsetpos(_file, &_start_feature_pos);	
	_is_new_feature = true;
}

int Cache::ReadVariable(feature_t *feature_id)
{
	if (_is_new_feature) {

		fgetpos(_file, &_start_feature_pos);

		if (!fread(feature_id, sizeof(feature_t), 1, _file)) {
			cout << "return 0" << endl;
			return 0;
		}
		_is_new_feature = false;
		return 1;
	}

	if (feof(_file))
		return 0;
}

void Cache::Rewind() 
{
	rewind(_file);
	_is_new_feature = true;
	fgetpos(_file, &_start_feature_pos);
}

void Cache::MoveToVar(feature_t feature_id)
{
	fsetpos(_file, &_start_var_pos[feature_id]);
	_is_new_feature = true;
}

int Cache::ReadLine(example_t *example_id, float *x, int *y, float *weight)
{
	if (_is_new_feature)
		return 0;	

	if (feof(_file))
		return 0;

	if (!fread(example_id, sizeof(example_t), 1, _file))
		return 0;
	if (!fread(x, sizeof(float), 1, _file))
		return 0;

	if ((*example_id == 0) && (*x == 0.0)) {
		_is_new_feature = true;
		return 0;
	}

	*y = _all_y[*example_id];
	//*weight = _weight[*example_id];
	*weight = 1;

	return 1;
}

void update_betaTx(float delta_beta)
{
	feature_t feature_id;
	example_t example_id;
	int y;
	float weight, x;
	
	g_cache.ReadVariable(&feature_id); 

	while(g_cache.ReadLine(&example_id, &x, &y, &weight)) {
		g_betaTx_delta[example_id] += delta_beta * x;
	}
}

double get_qloss()
{
	double qloss = 0.0;

	for (int i = 0; i < g_feature_count; i++) 
		qloss += g_lambda_1 * fabs(g_beta_new[i]);
	
	for (int i = 0; i < g_example_count; i++) {
		float exp_example_betaTx = g_exp_betaTx[i];
		float p = exp_example_betaTx / (1.0 + exp_example_betaTx);
		if (p < 1.0e-6)
			p = 1.0e-6;
		if (p > 0.999999)
			p = 0.999999;

		float w = p * (1 - p);
		int y01 = (g_all_y[i] + 1) / 2;

		qloss += 0.5 * w * SQUARE((y01 - p) / w - g_betaTx_delta[i]);
	}

	return qloss;
}

float* get_grad(const char *cache_filename)
{
	float *y_betaTx = (float*)safe_calloc(g_example_count, sizeof(float));
	float *grad = (float*)safe_calloc(g_feature_count, sizeof(float));
	feature_t feature_id;
	example_t example_id;
	int y;
	float weight, x; 

	g_cache.Rewind();
 
	while (g_cache.ReadVariable(&feature_id)) {
		while (g_cache.ReadLine(&example_id, &x, &y, &weight)) {
			y_betaTx[example_id] += y * g_beta[feature_id] * x;
		}
	}

	g_cache.Rewind();

	while (g_cache.ReadVariable(&feature_id)) {
		while (g_cache.ReadLine(&example_id, &x, &y, &weight)) {
			grad[feature_id] += - y * x / (1.0 + exp(y_betaTx[example_id]));
		}
	}

	for (int i = 0; i < g_feature_count; i++)
		grad[i] = soft_threshold(grad[i], g_lambda_1);

	safe_free(y_betaTx);

	return grad;
}

double get_loss_exact(const char *cache_filename, float alpha)
{
	start_timer("debug: calc loss exact");
	float *y_betaTx = (float*)safe_calloc(g_example_count, sizeof(float));
	feature_t feature_id;
	example_t example_id;
	int y;
	float weight, x; 

	g_cache.Rewind();

	while (g_cache.ReadVariable(&feature_id)) {
		while (g_cache.ReadLine(&example_id, &x, &y, &weight)) {
			y_betaTx[example_id] += y * ((1 - alpha) * g_beta[feature_id] + alpha * g_beta_new[feature_id])* x;
		}
	}

	accumulate_vector(g_master_location, y_betaTx, g_example_count);

	double loss = 0.0;

	for (int i = 0; i < g_example_count; i++) {
		float logit = -y_betaTx[i];
		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	safe_free(y_betaTx);

	for (int i = 0; i < g_feature_count; i++) {
		loss += g_lambda_1 * fabs((1 - alpha) * g_beta[i] + alpha * g_beta_new[i]);
	}

	stop_timer("debug: calc loss exact");
	return loss;
}

double get_grad_norm_exact(const char *cache_filename)
{
	start_timer("debug: calc gradient");
	double subgrad_norm = 0.0;
	float *subgrad = get_grad(cache_filename);

	for (int i = 0; i < g_feature_count; i++)
		subgrad_norm += SQUARE(subgrad[i]);

	safe_free(subgrad);
	stop_timer("debug: calc gradient");
	return subgrad_norm;
}

double get_loss(float alpha, float *reg_value)
{
	double loss = 0.0;

	for (int i = 0; i < g_feature_count; i++) {
		loss += g_lambda_1 * fabs((1 - alpha) * g_beta[i] + alpha * g_beta_new[i]);
	}

	if (reg_value)
		*reg_value = loss;

	for (int i = 0; i < g_example_count; i++) {
	
//		float log_odds = g_all_y[i] * (g_betaTx[i] + alpha * g_betaTx_delta[i]);
//		loss += (log_odds < -100 ? log_odds : log(1.0 + exp(-log_odds)));

		float exp_log_odds = (g_all_y[i] == -1 ? g_exp_betaTx[i] : 1.0 / g_exp_betaTx[i]);
		exp_log_odds *= exp(-g_all_y[i] * alpha * g_betaTx_delta[i]);

		loss += log(1.0 + exp_log_odds);
	}

	return loss;
}

void golden_section_search(double (*f)(float x), int count_max, float a0, float b0, float *xmin, double *fmin)
{
	float c = 0.3819;

	bool known_y = false;
	bool known_z = false;

	float a = a0, b = b0;
	float f_a = -100, f_y, f_z, f_b = -100;

	for (int i = 0; i < count_max; ++i) {	
		float y = a + c * (b - a);
		float z = a + (1 - c) * (b - a);

		if (!known_y) f_y = f(y);
		if (!known_z) f_z = f(z);

		//printf("a = %f y = %f z = %f b = %f\n", a, y, z, b);
		//printf("f_a = %f f_y = %f f_z = %f f_b = %f\n", f_a, f_y, f_z, f_b);

		if (f_y <= f_z) {
			*xmin = y;
			*fmin = f_y;

			//a = a;
			//y = new point
			b = z;
			z = y;

			//f_a = f_a;
			//f_y = ???;		
			f_b = f_z;
			f_z = f_y;

			known_y = false;
			known_z = true;
		}
		else {
			*xmin = z;
			*fmin = f_z;

			a = y;
			y = z;
			//z = new point
			//b = b;

			f_a = f_y;
			f_y = f_z;
			//f_z = ???
			//f_b = f_b;

			known_y = true;
			known_z = false;
		}

		printf("xmin = %f fmin = %e\n", *xmin, *fmin);
	}

	if (a == a0) {
		f_a = f(a);

		if (f_a < *fmin) {
			*xmin = a;
			*fmin = f(a);
		}
	}

	if (b == b0) {
		f_b = f(b);

		if (f_b < *fmin) {
			*xmin = b;
			*fmin = f(b);
		}
	}

	printf("xmin = %f fmin = %e\n", *xmin, *fmin);
}

void update_feature(float sum_w_q_x, float sum_w_x_2, float beta_max, feature_t feature_id, float shrinkage, float shrinkage_max, float beta_min, bool zero_max_shrinkage, vector<float> *feature_lambda)
{
	float beta_after_cd = 0.0;

	if (zero_max_shrinkage && (g_beta_new[feature_id] == 0.0))
		shrinkage = shrinkage_max;

	float A = sum_w_x_2 * shrinkage;
	float B = sum_w_q_x + (shrinkage - 1.0 + NU) * sum_w_x_2 * g_beta_new[feature_id];

	if (feature_lambda)
		feature_lambda->at(feature_id) = B;
		
	if (sum_w_x_2 != 0.0)
		beta_after_cd = soft_threshold(B, g_lambda_1) / A;
	else
		beta_after_cd = 0.0;

	if (fabs(beta_after_cd) > beta_max)
		beta_after_cd = SGN(beta_after_cd) * beta_max;
		
	float beta_step_max = 10.0;

	if (fabs(beta_after_cd - g_beta_new[feature_id]) > beta_step_max)
		if (beta_after_cd > g_beta_new[feature_id])
			beta_after_cd = g_beta_new[feature_id] + beta_step_max;
		else
			beta_after_cd = g_beta_new[feature_id] - beta_step_max;

	if (fabs(beta_after_cd) < beta_min)
		beta_after_cd = 0.0;

	g_cache.MoveToStartVariable();
	update_betaTx(beta_after_cd - g_beta_new[feature_id]);

	g_beta_new[feature_id] = beta_after_cd;
}

feature_t get_bad_coordinates(float beta_max)
{
	feature_t count = 0;

	for (int i = 0; i < g_feature_count; i++)
		count += (fabs(g_beta[i] - beta_max) < 1.0e-6);
	
	return count;
}


int file_exists(const char *filename)
{
	struct stat buf;
	int ret = stat(filename, &buf );

	return (ret == 0);
}

double f_alpha(float alpha) 
{ 
	return get_loss(alpha, NULL);
}


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
		float logit = log(g_exp_betaTx[i]);
		for (int k = 0; k < g_n; k++) 
			logit += alpha[k] * g_betaTx_delta_c[k][i];

		logit *= -g_all_y[i];

		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	return loss;
}

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

	if (type == 5) {
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
		int kcount = 30;
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
	buffer[0] = (char*)safe_calloc(1, sizeof(ulong));
	buffer[1] = (char*)safe_calloc(1, sizeof(ulong));

	get_kids_vectors(g_master_location, buffer, sizeof(ulong), global.unique_id, global.total, global.node, &child_sockets);

	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			ulong child_value = *((ulong*)buffer[k]);
			if (child_value > value)
				value = child_value;
		}
	}

	send_to_parent((char*)&value, sizeof(ulong));
	broadcast_buffer((char*)&value, sizeof(ulong));

	safe_free(buffer[0]);
	safe_free(buffer[1]);

	return value;
}

int is_less(double x1[], double x2[], int n)
{
	for (int i = 0; i < n; i++) {
		if (x1[i] < x2[i]) return 1;
		if (x1[i] > x2[i]) return 0;
	}

	return 1;
}

void min_allreduce(double x[], int n)
{
	int size = n * sizeof(double);

	char *buffer[2];
	int *child_sockets;
	buffer[0] = (char*)safe_calloc(1, size);
	buffer[1] = (char*)safe_calloc(1, size);

	get_kids_vectors(g_master_location, buffer, size, global.unique_id, global.total, global.node, &child_sockets);

	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			double *child_x = (double*)buffer[k];

			if (is_less(child_x, x, n))
				memcpy(x, child_x, size);
		}
	}

	safe_free(buffer[0]);
	safe_free(buffer[1]);

	send_to_parent((char*)x, size);
	broadcast_buffer((char*)x, size);
}

void print_vector(const char* name, float *v, int n)
{
	printf("%s:\n", name);

	for (int i = 0; i < n; i++)
		if (fabs(v[i]) > 1.0e-6)
			printf("%d: %f\n", i, v[i]);

	printf("\n");
}

void back_search(double zero_alpha_loss, float *best_alpha, double *min_loss, int *count, double *sum_loss)
{
	if (count) *count = 0;

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

		double exact_loss = 0.0; //get_loss_exact(g_cache_filename.c_str(), example_count, feature_count, lambda_1, alpha, beta, beta_new);

		//float *beta_zero = (float*)safe_calloc(feature_count, sizeof(float));
		double zero_beta_loss = 0.0; //get_loss_exact(g_cache_filename.c_str(), example_count, feature_count, lambda_1, alpha, beta_zero, beta_zero);
		//free(beta_zero);

		double loss = get_loss(alpha, NULL);
		if (count) (*count)++;

		if (sum_loss && i == 0) {
			*sum_loss = loss;			
		}
		
		printf("alpha % f loss %e exact_loss %e zero_beta_loss %e\n", alpha, loss, exact_loss, zero_beta_loss);

		if (loss < zero_alpha_loss) {
			*best_alpha = alpha;
			*min_loss = loss;
			return;
		}
	}

	*best_alpha = 0.0;
	*min_loss = zero_alpha_loss;
}

void back_search_armijo(double zero_alpha_loss, float alpha_init, float *subgrad, float *best_alpha, double *min_loss, int *count)
{
	double subgrad_L_delta_beta = 0.0;

	for (int i = 0; i < g_feature_count; ++i) {
		subgrad_L_delta_beta += subgrad[i] * (g_beta_new[i] - g_beta[i]);
	}

	if (count) *count = 0;

	*best_alpha = 0.0;
	*min_loss = zero_alpha_loss;

	float reg_value_new = 0.0, reg_value = 0.0;

	for (int i = 0; i < g_feature_count; ++i) {
		reg_value += g_lambda_1 * fabs(g_beta[i]);
		reg_value_new += g_lambda_1 * fabs(g_beta_new[i]);
	}

	for (int i = 0; i <= BACK_SEARCH_ARMIJO_MAX; i++) {
		double alpha = alpha_init * pow(0.5, i);
		double loss = get_loss(alpha, NULL);
		if (count) (*count)++;
		
		int armijo_ok = (int)((loss - zero_alpha_loss) < (SIGMA * alpha * (subgrad_L_delta_beta + reg_value_new - reg_value)));

		printf("armijo: count %d alpha %f loss %e ok %d\n", *count, alpha, loss, armijo_ok);

		if (armijo_ok) {
			*min_loss = loss;
			*best_alpha = alpha;
			break;			
		}
	}
}

void get_final_coeffs(double xmin[], float *coeffs)
{
	char *buffer_coeffs[2];
	int *child_sockets;
	buffer_coeffs[0] = (char*)safe_calloc(global.total, sizeof(float));
	buffer_coeffs[1] = (char*)safe_calloc(global.total, sizeof(float));

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

	safe_free(buffer_coeffs[0]);
	safe_free(buffer_coeffs[1]);
}

int myrandom (int i)
{
	return rand() % i;
}

float fmax(float *v, int count)
{
	float max_value = FLT_MIN;

	for (int i = 0; i < count; ++i) 
		if (v[i] > max_value)
			max_value = v[i];		

	return max_value;
}

float square_norm(float *v, int count)
{
	float sum = 0.0;

	for (int i = 0; i < count; ++i) 
		sum += v[i] * v[i];

	return sqrt(sum);
}

void combine_vectors(double loss, int type, float coeffs[], int random_count)
{
	if (type == 0 || type == 5 || type == 6) { // add all deltas

		if (type == 0 || type == 6) {   // simple sum, shrinked sum
			for (int i = 0; i < global.total; ++i)
				coeffs[i] = 1.0;
		}

		if (type == 5) {  // quasi-gradient step in hypercoordinates
			float best_alpha;
			double local_min_loss;

			start_timer("combine vectors - back search");
			back_search(loss, &best_alpha, &local_min_loss, NULL, NULL);
			printf("Local: prev_loss  %e\n", loss);
			printf("Local: best_alpha %.4f local_min_loss %e node %ld\n", best_alpha, local_min_loss, global.node);
			stop_timer("combine vectors - back search");

			float weights[global.total];
			memset(weights, 0, global.total * sizeof(float));
			weights[global.node] = loss - local_min_loss;
			accumulate_vector(g_master_location, weights, global.total);

			print_coeffs_style("node weights", "%e ", weights, global.total);

			float norm = square_norm(weights, global.total);

			if (norm > 0 ) {
				for (int i = 0; i < global.total; ++i) {
					coeffs[i] = weights[i] / norm;
				}
			}
			
			print_coeffs("node coeffs unnormed", coeffs, global.total);

			float max_coeff = fmax(coeffs, global.total);

			if (max_coeff > 0) {
				for (int i = 0; i < global.total; ++i)
					coeffs[i] /= max_coeff;			
			}

			print_coeffs("node coeffs", coeffs, global.total);

			for (int i = 0; i < g_example_count; ++i)
				g_betaTx_delta[i] *= best_alpha * coeffs[global.node];

			for (int i = 0; i < g_feature_count; ++i)
				g_beta_new[i] = g_beta[i] + (g_beta_new[i] - g_beta[i]) * best_alpha * coeffs[global.node];
		}

		start_timer("combine vectors - data transfer");
		accumulate_vector(g_master_location, g_betaTx_delta, g_example_count);
		accumulate_vector(g_master_location, g_beta_new, g_feature_count);
		stop_timer("combine vectors - data transfer");
	
		for (int i = 0; i < g_feature_count; i++)
			g_beta_new[i] -= (global.total - 1) * g_beta[i];

		return;
	}
	else if (type == 4) { // random

		vector<int> nodes(global.total);

		for (int i = 0; i < global.total; ++i) {
			nodes[i] = i;
			coeffs[i] = 0.0;
		}

		random_shuffle(nodes.begin(), nodes.end(), myrandom);

		for (int i = 0; i < random_count; ++i) {
			coeffs[nodes[i]] = 1.0;
		}

		if (coeffs[global.node] == 0.0) {
			memset(g_betaTx_delta, 0, g_example_count * sizeof(float));
			memset(g_beta_new, 0, g_feature_count * sizeof(float));
		}

		start_timer("combine vectors - data transfer");
		accumulate_vector(g_master_location, g_betaTx_delta, g_example_count);
		accumulate_vector(g_master_location, g_beta_new, g_feature_count);
		stop_timer("combine vectors - data transfer");

		for (int i = 0; i < g_feature_count; i++)
			g_beta_new[i] -= (random_count - 1) * g_beta[i];

		return;
	}
	else if (type == 1) {  // select best

		float best_alpha;
		double min_loss;
		memset(coeffs, 0, global.total * sizeof(float));

		start_timer("combine vectors - back search");
		back_search(loss, &best_alpha, &min_loss, NULL, NULL);
		stop_timer("combine vectors - back search");

		start_timer("combine vectors - find best");
		float node_losses[global.total];
		memset(node_losses, 0, global.total * sizeof(float));
		node_losses[global.node] = min_loss;

		accumulate_vector(g_master_location, node_losses, global.total);

		int best_node = 0;
		float best_loss = node_losses[0];

		for (int i = 0; i < global.total; ++i) 	{
			if (node_losses[i] < best_loss) {
				best_node = i;
				best_loss = node_losses[i];
			}
		}		

		printf("Local: min_loss %e node %ld\n", min_loss, global.node);
		printf("Best : min_loss %e node %d\n", best_loss, best_node);
		stop_timer("combine vectors - find best");

		if (best_node != global.node) {
			memset(g_betaTx_delta, 0, g_example_count * sizeof(float));
			memset(g_beta_new, 0, g_feature_count * sizeof(float));
			printf("debug: I am deleted...\n");
		}
		else {
			coeffs[global.node] = 1.0;
			printf("debug: I am alive!\n");
		}

		start_timer("combine vectors - data transfer");
		accumulate_vector(g_master_location, coeffs, global.total);
		accumulate_vector(g_master_location, g_betaTx_delta, g_example_count);
		accumulate_vector(g_master_location, g_beta_new, g_feature_count);
		stop_timer("combine vectors - data transfer");
		return;
	}

	//
	// Read kids' vectors
	//
	ulong reduce_vector_count = g_example_count + g_feature_count;
	char *buffer[2];
	int *child_sockets;
	buffer[0] = (char*)safe_calloc(reduce_vector_count, sizeof(float));
	buffer[1] = (char*)safe_calloc(reduce_vector_count, sizeof(float));

	start_timer("combine vectors - data transfer");
	get_kids_vectors(g_master_location, buffer, reduce_vector_count * sizeof(float), global.unique_id, global.total, global.node, &child_sockets);
	stop_timer("combine vectors - data transfer");

	//
	// Prepare local vector
	//
	start_timer("combine vectors - local opt");
	float *full_vector = (float*)safe_calloc(reduce_vector_count, sizeof(float));

	memcpy(full_vector, g_betaTx_delta, g_example_count * sizeof(float));

	for (int i = 0; i < g_feature_count; i++)
		full_vector[g_example_count + i] = g_beta_new[i] - g_beta[i];

	//print_full_vectors(full_vector, reduce_vector_count, buffer, child_sockets);

	g_betaTx_delta_c[0] = full_vector;
	g_beta_delta[0] = full_vector + g_example_count;

	int n = 1;
	for (int k = 0; k < 2; k++) {
		if (child_sockets[k] != -1) {
			g_betaTx_delta_c[n] = (float*)(buffer[k]);
			g_beta_delta[n] = ((float*)(buffer[k]) + g_example_count);
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

	memcpy(g_betaTx_delta, full_vector, g_example_count * sizeof(float));

	for (int i = 0; i < g_feature_count; i++)
		g_beta_new[i] = g_beta[i] + full_vector[g_example_count + i];

	safe_free(full_vector);
	safe_free(buffer[0]);
	safe_free(buffer[1]);

	//
	// Calculating final linear coeffs
	//
	memset(coeffs, 0, sizeof(float) * global.total);
	coeffs[global.node] = (n > 1 ? xmin[0] : 1.0);
	get_final_coeffs(xmin, coeffs);
}

void save_beta(string filename)
{
	start_timer("saving beta");
	FILE *file_rfeatures = fopen(filename.c_str(), "w");

	fprintf(file_rfeatures, "solver_type L1R_LR\n");
	fprintf(file_rfeatures, "nr_class 2\n");
	fprintf(file_rfeatures, "label 1 -1\n");
	fprintf(file_rfeatures, "nr_feature %ld\n", (ulong)(g_feature_count - 1));
	fprintf(file_rfeatures, "bias -1\n");
	fprintf(file_rfeatures, "w\n");

	float tolerance = 0.0; // 1.0e-6

	for (int i = 1; i < g_feature_count; i++) {
		if (fabs(g_beta[i]) <= tolerance) {
			fprintf(file_rfeatures, "0\n", g_beta[i]);	
		}
		else {
			fprintf(file_rfeatures, "%f\n", g_beta[i]);	
		}
	}

	fclose(file_rfeatures);
	stop_timer("saving beta");
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

int get_work_time()
{
	int time_total = 0;
	for (map<string, int>::iterator it = g_timers.begin(); it != g_timers.end(); ++it) {
		if (!strstr(it->first.c_str(), "debug"))
			time_total += it->second;
	}
	
	return time_total;
}

int get_full_time()
{
	int time_total = 0;
	for (map<string, int>::iterator it = g_timers.begin(); it != g_timers.end(); ++it) {
		time_total += it->second;
	}
	
	return time_total;
}

class IterStat
{
public:
	double loss;
	int time;
	int full_time;
	double subgrad_norm_local;
	double subgrad_norm;
	double coeffs_norm;
	feature_t global_bad_coord;
	double alpha;
	int back_search_count;
	float rel_loss_diff;
	float ideal_rel_loss_diff;
	float rel_sum_err;
};

double delta_norm(float *v1, float *v2, feature_t feature_count)
{
	double norm = 0.0;
	
	for (int i = 0; i < feature_count; i++) {
		norm += SQUARE(v2[i] - v1[i]);
	}

	return norm;
}

float newton_step(float bias, int *all_y, float lambda_1, example_t example_count)
{
	double sum_w_x_2 = 0;
	double sum_w_q_x = 0;

	for (int i = 0; i < example_count; ++i)
	{
		float example_betaTx = bias;
		float exp_example_betaTx = exp(example_betaTx);
		float exp_y_example_betaTx = exp_example_betaTx;

		int y = all_y[i];

		if (y == -1)
			exp_y_example_betaTx = 1.0 / exp_y_example_betaTx;				

		float p = exp_example_betaTx / (1.0 + exp_example_betaTx);
		if (p < 1.0e-6)
			p = 1.0e-6;
		if (p > 1.0 - 1.0e-6)
			p = 1.0 - 1.0e-6;

		float w = p * (1 - p);
		int y01 = (y + 1) / 2;
		float z = example_betaTx + (y01 - p) / w;
		float q = z - (example_betaTx - bias);

		sum_w_x_2 += w * 1 * 1;
		sum_w_q_x += w * q * 1;
	}

	if (sum_w_x_2 != 0.0)
		bias = soft_threshold(sum_w_q_x, lambda_1) / sum_w_x_2;
	else
		bias = 0.0;

	return bias;
}

float find_bias(int steps_max, int *all_y, float lambda_1, example_t example_count)
{
	float bias = 0.0;

	for (int i = 0; i < steps_max; ++i) {
		bias = newton_step(bias, all_y, lambda_1, example_count);
		printf("newton step %d bias = %f\n", i + 1, bias);
	}

	return bias;
}
 
void get_feature_lambda(float *f_lambda)
{
	g_cache.Rewind();

	for (feature_t feature_idx = 0; feature_idx < g_cache.GetCacheFeatureCount(); ++feature_idx) {

		feature_t feature_id = g_cache.GetFeatureId(feature_idx);

		g_cache.ReadVariable(&feature_id);

		example_t example_id;
		int y;
		float weight, x;

		double sum_w_x_2 = 0.0;
		double sum_w_q_x = 0.0;

		while (g_cache.ReadLine(&example_id, &x, &y, &weight)) {

			float exp_example_betaTx = g_exp_betaTx[example_id];

			float exp_y_example_betaTx = (y == 1 ? exp_example_betaTx : 1.0 / exp_example_betaTx); 
			float p = exp_example_betaTx / (1.0 + exp_example_betaTx);

			p = LIMIT(p, P_MIN, P_MAX);

			float w = p * (1 - p);
			int y01 = (y + 1) / 2;

			float q = (y01 - p) / w - (g_betaTx_delta[example_id] - g_beta_new[feature_id] * x);

			sum_w_x_2 += w * x * x;
			sum_w_q_x += w * q * x;
		}

		f_lambda[feature_id] = sum_w_q_x;
	}
}

void read_beta(bool distributed, const char *filename, float *g_beta, float* g_beta_new, float *g_exp_betaTx)
{
	float *betaTx = (float*)safe_calloc(g_example_count, sizeof(float));

	//
	// Read beta from file
	//
	FILE *file = fopen(filename, "r");

	for (int i = 0; i < 6; ++i) {
		char *tmp = fgets(g_buffer, sizeof(g_buffer), file);		
	}

	feature_t feature_id = 1;

	while (fgets(g_buffer, sizeof(g_buffer), file)) {
		g_beta[feature_id] = g_beta_new[feature_id] = atof(g_buffer);
		feature_id++;
	}

	fclose(file);

	//
	// Calculate exp(betaTx)
	//
	g_cache.Rewind();

	for (feature_t feature_idx = 0; feature_idx < g_cache.GetCacheFeatureCount(); ++feature_idx) {

		feature_t feature_id = g_cache.GetFeatureId(feature_idx);

		g_cache.ReadVariable(&feature_id);

		example_t example_id;
		int y;
		float weight, x;

		while (g_cache.ReadLine(&example_id, &x, &y, &weight)) {
			betaTx[example_id] += g_beta[feature_id] * x;
		}
	}

	if (distributed) {
		accumulate_vector(g_master_location, betaTx, g_example_count);
	}

	for (example_t i = 0; i < g_example_count; ++i) {
		g_exp_betaTx[i] = exp(betaTx[i]);
	}

	safe_free(betaTx);
}

void optimize(int iterations_max, int lambda_idx, const po::variables_map& vm, string cache_filename, float termination_eps)
{
	float beta_max = vm["beta-max"].as<float>();
	bool distributed = !vm["server"].empty();
	int combine_type = vm["combine-type"].as<int>();

	float sum_coeffs[global.total];
	memset(sum_coeffs, 0, sizeof(float) * global.total);
	
	double sum_loss;
	double prev_loss = get_loss(0.0, NULL);
	printf("loss %f\n", prev_loss);
	double prev_qloss = get_qloss();
	printf("qloss %f\n", prev_qloss);
	float shrinkage = vm["initial-shrinkage"].as<float>();
	float shrinkage_max = shrinkage;

	IterStat iter_stat[iterations_max + 1];
	memset(iter_stat, 0, sizeof(iter_stat));
	iter_stat[0].loss = prev_loss;
	iter_stat[0].time = 0;
//	iter_stat[0].subgrad_norm_local = get_grad_norm_exact(cache_filename.c_str());
	iter_stat[0].subgrad_norm_local = 0.0;

	int active_feature_iter = 0;
	int active_count = 0;
	int cd_count = 0;

	vector<char> active(g_feature_count, 1);

	float *subgrad = (float*)safe_calloc(g_feature_count, sizeof(float));
	int iterations_done;

	printf("\n");
	printf("lambda_1 = %f\n", g_lambda_1);
	printf("\n");

	for (int iter = 1; iter <= iterations_max; iter++) {
		
		start_timer("iterations");
		printf("Iteration %d\n", iter);
		printf("--------------\n");
		printf("active %d\n", active_feature_iter);

		feature_t processed_features = 0;
		int lines_processed = 0;
		memset(subgrad, 0, g_feature_count * sizeof(float));

		if (!active_feature_iter) {
			fill(active.begin(), active.end(), 1);
			prev_qloss = get_qloss();
		}

		g_cache.Rewind();

		for (feature_t feature_idx = 0; feature_idx < g_cache.GetCacheFeatureCount(); ++feature_idx) {

			feature_t feature_id = g_cache.GetFeatureId(feature_idx);

			if (!active[feature_id]) {
				continue;
			}
			else {
				g_cache.MoveToVar(feature_id);
				processed_features++;
			}

			g_cache.ReadVariable(&feature_id);

			example_t example_id;
			int y;
			float weight, x;

			double sum_w_x_2 = 0.0;
			double sum_w_q_x = 0.0;

			while (g_cache.ReadLine(&example_id, &x, &y, &weight)) {

				float exp_example_betaTx = g_exp_betaTx[example_id];

				float exp_y_example_betaTx = (y == 1 ? exp_example_betaTx : 1.0 / exp_example_betaTx); 
				float p = exp_example_betaTx / (1.0 + exp_example_betaTx);

				p = LIMIT(p, P_MIN, P_MAX);

				float w = p * (1 - p);
				int y01 = (y + 1) / 2;
				//float z = example_betaTx + (y01 - p) / w;
				//float q = z - (example_betaTx + betaTx_delta[example_id] - beta_new[feature_id] * x);

				float q = (y01 - p) / w - (g_betaTx_delta[example_id] - g_beta_new[feature_id] * x);

				sum_w_x_2 += w * x * x;
				sum_w_q_x += w * q * x;

				subgrad[feature_id] += - y * x / (1.0 + exp_y_example_betaTx);
				
				lines_processed++;
			}
		
			//vector<float> *f_lambda = ((iter == 1) ? &feature_lambda : NULL);
			vector<float> *f_lambda = NULL;
				
			update_feature(sum_w_q_x, sum_w_x_2, beta_max, feature_id, shrinkage, shrinkage_max,
					vm["beta-min"].as<float>(), vm["zero-max-shrinkage"].as<int>(), f_lambda);
		}

		stop_timer("iterations");

		double min_loss;
		float best_alpha;
	
		printf("processed features total %d\n", processed_features);
		printf("lines processed %d\n", lines_processed);

		start_timer("debug: calc bad coordinates");
		printf("\n");	
		printf("Local ||beta - beta_new|| = %e\n", delta_norm(g_beta, g_beta_new, g_feature_count));
		feature_t bad_coord = get_bad_coordinates(beta_max);
		if (distributed) {
			printf("bad coordinates %ld\n", (ulong)bad_coord);
			iter_stat[iter].global_bad_coord = (feature_t)(accumulate_scalar(g_master_location, bad_coord) + 0.5);
		}
		stop_timer("debug: calc bad coordinates");

		//
		// Accumulate deltas of beta
		//
		if (distributed) {
			if (vm["sync"].as<int>()) sync_nodes();

			//start_timer("deltas combination");

			float coeffs[global.total];
			combine_vectors(prev_loss, combine_type, coeffs, vm["random-count"].as<int>());

			printf("\n");
			float c_abs_norm = 0.0;
			for (int i = 0; i < global.total; i++) {
				printf("%5d ", i);
				c_abs_norm += fabs(coeffs[i]);
				sum_coeffs[i] += fabs(coeffs[i]);
			}

			printf("\n");
			print_coeffs("coeffs", coeffs, global.total);
			print_coeffs("sum_coeffs", sum_coeffs, global.total);

			printf("||c||_1 = %f\n", c_abs_norm);
			printf("\n");

			iter_stat[iter].coeffs_norm = c_abs_norm;
		
			printf("network time, sec %f \n", get_comm_time() * 1.0e-3);
			//stop_timer("deltas combination");
		}

		start_timer("debug: calc subgradient");
		double subgrad_norm_local = 0.0, subgrad_norm;

		for (int i = 0; i < g_feature_count; i++) {
			subgrad[i] = soft_threshold(subgrad[i], g_lambda_1);
			subgrad_norm_local += SQUARE(subgrad[i]);
		}
	
		if (distributed) {
			subgrad_norm = accumulate_scalar(g_master_location, subgrad_norm_local);
		}
		else {	
			subgrad_norm = subgrad_norm_local;
		}

		stop_timer("debug: calc subgradient");

		double subgrad_norm_exact_local = 0.0, subgrad_norm_exact = 0.0;
		iter_stat[iter].subgrad_norm_local = subgrad_norm_local;
		iter_stat[iter].subgrad_norm = subgrad_norm;

		//
		// Linear search
		//
		if (!vm.count("no-back-search")) {
			start_timer("linear search");
			printf("Testing combined delta:\n");

			if (!vm.count("linear-search")) {
				back_search_armijo(prev_loss, 1.0, subgrad, &best_alpha, &min_loss, &(iter_stat[iter].back_search_count));

				sum_loss = get_loss(1.0, NULL);
			}
			else {
				float xmin;
				double fmin;

				double f_0 = f_alpha(0);
				double f_1 = f_alpha(1.0);
				sum_loss = f_1;
				iter_stat[iter].rel_sum_err = (f_1 - f_0) / f_0;

				printf("(f_1 - f_0) / f_0 = %e\n", iter_stat[iter].rel_sum_err);

				if (f_1 < f_0) {
					xmin = 1.0;
					fmin = f_1;
				}
				else {
					golden_section_search(f_alpha, 10, 0, 1, &xmin, &fmin);
				}

				back_search_armijo(prev_loss, xmin, subgrad, &best_alpha, &min_loss, &(iter_stat[iter].back_search_count));
			}
			printf("\n");
			stop_timer("linear search");

			if (combine_type == 6) {
				
				shrinkage_max = max(shrinkage, shrinkage_max);

				printf("shrinkage %f\n", shrinkage);
				printf("shrinkage_max %f\n", shrinkage_max);

				if (best_alpha < 1.0 && vm["increase-shrinkage"].as<int>()) {
					shrinkage *= 2; 
				}

				if (best_alpha == 1.0 && vm["decrease-shrinkage"].as<int>()) {
					shrinkage /= 2;
				}
			}
		}
		else {
			best_alpha = 1.0;
			min_loss = get_loss(1.0, NULL);
			sum_loss = min_loss;
			iter_stat[iter].back_search_count = 0;			
		}

		//
		// Update counters
		//
		int make_newton;
		int cd_max = 1;
		cd_count++;

		if (cd_count < cd_max) {
			make_newton = 0;
		}
		else {
			make_newton = 1;
			cd_count = 0;
		}
	
		//
		// Stoping criteria
		//	
		double ideal_rel_loss_diff = ((min_loss - prev_loss) / prev_loss);
		bool stop = (iter == iterations_max) || ((fabs(ideal_rel_loss_diff) < termination_eps) && (sum_loss <= prev_loss * (1 + termination_eps)));

		//
		// Maybe change alpha ?
		//
		if (vm.count("last-iter-sum") && stop) {
			best_alpha = 1.0;
		}

		//
		// Update stats
		//
		double loss_new = get_loss(best_alpha, NULL);
		double qloss_new = get_qloss();
		double rel_loss_diff = ((loss_new - prev_loss) / prev_loss);
		double rel_qloss_diff = ((qloss_new - prev_qloss) / prev_qloss);

		printf("loss %e min_loss %e qloss %e ||subgrad||_2 %e rel_loss_diff %e rel_qloss_diff %e\n", loss_new, min_loss, qloss_new, subgrad_norm, rel_loss_diff, rel_qloss_diff);
		cout << "iter " << iter << " CD best_alpha " << best_alpha << " min_loss " << min_loss << "\n";

		iter_stat[iter].loss = loss_new;
		iter_stat[iter].rel_loss_diff = rel_loss_diff;
		iter_stat[iter].ideal_rel_loss_diff = ideal_rel_loss_diff;
		iter_stat[iter].time = get_work_time();
		iter_stat[iter].full_time = get_full_time();
		iter_stat[iter].alpha = best_alpha;

		iterations_done = iter;

		//
		// Make step
		//
		if (make_newton) {
			for (int i = 0; i < g_example_count; i++) {
				//g_betaTx[i] += best_alpha * g_betaTx_delta[i];
				//g_betaTx[i] = LIMIT(g_betaTx[i], -MAX_betaTx, MAX_betaTx);
				//g_exp_betaTx[i] = exp(g_betaTx[i]);

				g_exp_betaTx[i] *= exp(best_alpha * g_betaTx_delta[i]);
			
				g_betaTx_delta[i] = 0.0;
			}

			for (int i = 0; i < g_feature_count; i++) {
				g_beta[i] = (1 - best_alpha) * g_beta[i] + best_alpha * g_beta_new[i];
				g_beta_new[i] = g_beta[i];
			}
			cout << "iter " << iter << " NW best_alpha " << best_alpha << " min_loss " << min_loss << "\n";
		}

		print_time();

		if (vm.count("save-per-iter") && !vm["final-regressor"].empty()) {
			string filename = vm["final-regressor"].as<string>() + string(".") + to_string(lambda_idx, "%03d") + string(".") + to_string(iter, "%03d");
			cout << "saving regressor into " << filename << endl;
			save_beta(filename);
		}

		if (make_newton && stop) {
			/*start_timer("debug: calc gradient");
			double grad_norm2 = get_grad_norm_exact(cache_filename.c_str(), example_count, feature_count, lambda_1, beta_new);
			if (distributed)
				grad_norm2 = accumulate_scalar(g_master_location, grad_norm2);
			printf("||grad||_2 = %e\n", grad_norm2);
			stop_timer("debug: calc gradient");*/
			break;
		}

		if (make_newton) {
			prev_loss = loss_new;
		}
		prev_qloss = qloss_new;

		cout << "\n";

		//
		// Active iterations handling
		//
		if (!active_feature_iter && vm.count("active")) {
			for (int i = 0; i < g_feature_count; ++i) {
				if (fabs(g_beta_new[i]) < 1.0e-6) {
					active[i] = 0;
				}
			}

			active_feature_iter = 1;
			active_count = 0;
		}

		if (active_feature_iter)
			active_count++;

		if (active_count == 3) {
			active_count = 0;
			active_feature_iter = 0;
		}
	}

	safe_free(subgrad);

	printf("\n");	
	print_time_summary();

	if (!vm["final-regressor"].empty())
		save_beta(vm["final-regressor"].as<string>());

	//
	// Print iter-wise stat
	//
	printf("Iter\tLoss\tRelLossDiff\tIdealRelLossDiff\tRelSumErr\tTime\tFullTime\tSubGradNorm\tSubGradNormLocal\tCoeffsNorm\tGlobalBadCoord\tAlpha\tBackSearch\n");

	for (int i = 0; i <= iterations_done; ++i) {
		printf("%3d\t%.10e\t% .10e\t% .10e\t% .10e\t%d\t%d\t%.10e\t%.10e\t%f\t%ld\t%f\t%d\n",
			i, iter_stat[i].loss, iter_stat[i].rel_loss_diff, iter_stat[i].ideal_rel_loss_diff, iter_stat[i].rel_sum_err, 
			iter_stat[i].time, iter_stat[i].full_time, iter_stat[i].subgrad_norm, iter_stat[i].subgrad_norm_local, \
			iter_stat[i].coeffs_norm, (ulong)iter_stat[i].global_bad_coord, iter_stat[i].alpha, iter_stat[i].back_search_count);
	}

}

int main(int argc, char **argv)
{
	time(&g_start_time);
	srand(107);

	vector<float> default_lambda(1, 1);

	po::options_description general_desc("General options");
	general_desc.add_options()
        	("help,h", "produce help message")
        	("dataset,d", po::value<string>(), "training set in the inverted index form")
        	("labels,l", po::value<string>(), "labels of the training set")
        	("initial-regressor,i", po::value<string>(), "initial weights")
        	("final-regressor,f", po::value<string>(), "final weights")
		("lambda-1", po::value<vector<float> >()->default_value(default_lambda, "1.0"), "L1 regularization, allowed multiple values")
		("lambda-path", po::value<int>(), "L1 regularization, number of labmdas in regularization path")
		("combine-type,c", po::value<int>()->default_value(1), "type of deltas combination during AllReduce: \n0 - sum, 1 - best, 2 - Nelder-Mead, 3 - discrete opt, 4 - random, 5 - smart sum, 6 - shrinked sum")
		("termination", po::value<float>()->default_value(1.0e-4), "termination criteria")
		("iterations", po::value<int>()->default_value(100), "maximum number of iterations")
		("save-per-iter", "save beta every iteration")
		("beta-max", po::value<float>()->default_value(10.0), "maximum absolute beta[i]")
		("beta-min", po::value<float>()->default_value(0.0), "minimum absolute beta[i]")
		("random-count", po::value<int>()->default_value(2), "number of deltas taken randomly")
		("no-back-search", "don't make back search")
		("active", "use active features set")
		("initial-shrinkage", po::value<float>()->default_value(1.0), "initial shrinkage value")
		("increase-shrinkage", po::value<int>()->default_value(0), "increase shrinkage if alpha < 1.0")
		("decrease-shrinkage", po::value<int>()->default_value(0), "decrease shrinkage if alpha = 1.0")
		("zero-max-shrinkage", po::value<int>()->default_value(0), "always use maximum shrinkage for zero features")
		("linear-search", "make accurate linear search")
		("find-bias", "initial finding bias")
		("last-iter-sum", "add all differences at last iteration")
        	;

	po::options_description cluster_desc("Cluster options");
	cluster_desc.add_options()
        	("server", po::value<string>(), "master server")
        	("unique-id", po::value<int>()->default_value(1), "unique id of the learning transaction")
        	("node", po::value<int>(), "node index")
        	("total", po::value<int>(), "total number of nodes")
        	("sync", po::value<int>()->default_value(0), "sync nodes")
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

//	printf("%d", (int)vm.count("lambda-path"));
//	return 0;

	string dataset_filename = vm["dataset"].as<string>();
	vector<float> lambda_1 = vm["lambda-1"].as<vector<float> >();
	int iterations_max = vm["iterations"].as<int>();
	bool distributed = !vm["server"].empty();
	float termination_eps = vm["termination"].as<float>();

	extern global_data global;

	if (distributed) {
		g_master_location = vm["server"].as<string>();
		global.unique_id = vm["unique-id"].as<int>();
		global.node = vm["node"].as<int>();
		global.total = vm["total"].as<int>();
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

	feature_t max_feature_id = 0;
	example_t max_example_id = 0;
	ulong lines_count = 0;
	feature_t unique_features = 0;

	g_cache.Create(dataset_filename.c_str(), cache_filename.c_str(), &max_feature_id, &lines_count, &unique_features);
//	g_cache.InitY(g_example_count, g_feature_count, distributed, vm);
	g_cache.ReadY(vm["labels"].as<string>().c_str(), &max_example_id);

	printf("debug: max_feature_id %ld\n", (ulong)max_feature_id);
	printf("debug: max_example_id %ld\n", (ulong)max_example_id);

	if (distributed) {
		if (vm["sync"].as<int>()) sync_nodes();

		start_timer("sync dataset info");
		max_feature_id = max_allreduce(max_feature_id);
		stop_timer("sync dataset info");
	}

	g_feature_count = max_feature_id + 1;
	g_example_count = max_example_id + 1;

	g_cache.InitCache(g_feature_count);
	g_all_y = g_cache.GetY();

	printf("lines count      = %ld\n", lines_count);
	printf("unique features  = %ld\n", (ulong)unique_features);
	printf("feature count    = %ld\n", (ulong)g_feature_count);
	printf("example count    = %ld\n", (ulong)g_example_count);
	printf("final regressor  = %s\n", !vm["final-regressor"].empty() ? vm["final-regressor"].as<string>().c_str() : "");
	printf("combination type = %d\n", vm["combine-type"].as<int>());
	printf("iterations_max   = %d\n", iterations_max);
	printf("termination_eps  = %e\n", termination_eps);
	printf("random count     = %d\n", vm["random-count"].as<int>());
	printf("back search      = %d\n", !vm.count("no-back-search"));

	if (distributed) {
		printf("server = %s\n", g_master_location.c_str());
		printf("node = %ld\n", global.node);
		printf("total = %ld\n", global.total);
		printf("unique id = %ld\n", global.unique_id);
	}

	if (vm.count("initial-regressor"))
		printf("using initial regressor %s\n", vm["initial-regressor"].as<string>().c_str());

	printf("\n");
	printf("\n");

	//
	// Allocating important vectors
	//
	//g_betaTx = (float*)safe_calloc(g_example_count, sizeof(float));
	g_exp_betaTx = (float*)safe_calloc(g_example_count, sizeof(float));
	g_betaTx_delta = (float*)safe_calloc(g_example_count, sizeof(float));
	g_beta = (float*)safe_calloc(g_feature_count, sizeof(float));
	g_beta_new = (float*)safe_calloc(g_feature_count, sizeof(float));

	double avg_y = 0.0;

	for (int i = 0; i < g_example_count; ++i)
		avg_y += g_all_y[i];

	printf("avg y = %f\n", avg_y / g_example_count);

	if (vm.count("initial-regressor")) {
		read_beta(distributed, vm["initial-regressor"].as<string>().c_str(), g_beta, g_beta_new, g_exp_betaTx);	
	}
	else {		
		//
		// Find bias ?
		//
		float bias = 0.0;

		if (vm.count("find-bias")) {
			bias = find_bias(5, g_all_y, lambda_1[0], g_example_count);
			g_beta[1] = g_beta_new[1] = bias;
		}

		for (int i = 0; i < g_example_count; i++) {
			//g_betaTx[i] = bias;
			g_exp_betaTx[i] = exp(bias);
		}
	}

	print_time();

	//
	// Calc feature lambda
	// 
	float *feature_lambda = (float*)safe_calloc(g_feature_count, sizeof(float));

	if (vm.count("lambda-path")) {

		get_feature_lambda(feature_lambda);
		if (distributed) {
			accumulate_vector(g_master_location, feature_lambda, g_feature_count);
		}

		float lambda_max = 0.0;
		for (int i = 0; i < g_feature_count; ++i) {
			if (abs(feature_lambda[i]) > lambda_max) {
				lambda_max = abs(feature_lambda[i]);
			}
		}

		lambda_1.resize(vm["lambda-path"].as<int>());

		for (int i = 0; i < vm["lambda-path"].as<int>(); ++i) {
			lambda_1[i] = lambda_max * pow(2, - i - 1);	
		}
	}

	//
	// Real work is done here
	//
	for (int i = 0; i < lambda_1.size(); ++i) {
		g_lambda_1 = lambda_1[i];
		optimize(iterations_max, i, vm, cache_filename, termination_eps);
	}

	//
	// Write lambda for features
	//
	FILE *file_rfeatures = fopen("lambda_model", "w");

	for (int i = 1; i < g_feature_count; i++) {
		fprintf(file_rfeatures, "%d\t%e\n", i, feature_lambda[i]);
	}

	fclose(file_rfeatures);

	//safe_free(g_betaTx);
	safe_free(g_exp_betaTx);
	safe_free(g_betaTx_delta);
	safe_free(g_beta);  
	safe_free(g_beta_new);
	safe_free(feature_lambda);

	return 0;
}
