#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<climits>
#include<iostream>
#include<cfloat>

using std::cout;

typedef unsigned long int ulong_t;

static inline float soft_threshold(float x, float a) {
	if (fabs(x) < a)
		return 0.0;
	else if (x > 0.0)
		return x - a;
	else
		return x + a;
}

char g_buffer[1<<20];

int parse_line(FILE *file, ulong_t *feature_id, ulong_t *example_id, int *y, float *weight, float *x, fpos_t *fpos) {

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

void update_betaTx(FILE *file, int count, float delta_beta, float *betaTx)
{
	for (int i = 0; i < count; i++) {
		ulong_t feature_id, example_id;
		int y;
		float weight, x; 

		if (parse_line(file, &feature_id, &example_id, &y, &weight, &x, NULL))
			betaTx[example_id] += delta_beta * x;
	}
}

float get_qloss(float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y)
{
	float loss = 0.0;

	for (int i = 1; i <= feature_count; i++) 
		loss += lambda_1 * fabs(beta_new[i]);

	for (int i = 1; i <= example_count; i++) {
		float example_betaTx = betaTx[i];
		float p = 1.0 / (1.0 + exp(-example_betaTx));
		if (p < 1.0e-6)
			p = 1.0e-6;
		if (p > 0.999999)
			p = 0.999999;

		float w = p * (1 - p);
		int y01 = (all_y[i] + 1) / 2;
		float z = example_betaTx + (y01 - p) / w;

		loss += w * (z - betaTx[i] - betaTx_delta[i]) * (z - betaTx[i] - betaTx_delta[i]);
	}

	return loss;
}

float* get_grad(char *dataset_filename, ulong_t example_count, ulong_t feature_count, float lambda_1, float *beta)
{
	float *y_betaTx = (float*)calloc(example_count, sizeof(float));
	float *grad = (float*)calloc(feature_count, sizeof(float));
	ulong_t feature_id, example_id;
	int y;
	float weight, x; 

	FILE *file = fopen(dataset_filename, "r");

	while (parse_line(file, &feature_id, &example_id, &y, &weight, &x, NULL)) {
		y_betaTx[example_id] += y * beta[feature_id] * x;
	}

	rewind(file);

	while (parse_line(file, &feature_id, &example_id, &y, &weight, &x, NULL)) {
		grad[feature_id] += - y * x / (1.0 + exp(y_betaTx[example_id]));
	}

	for (int i = 0; i < feature_count; i++)
		grad[i] = soft_threshold(grad[i], lambda_1);

	fclose(file);
	free(y_betaTx);

	return grad;
}

float get_loss(float alpha, float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y) {

	float loss = 0.0;

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

void update_feature(FILE *file, ulong_t count, float sum_w_q_x, float sum_w_x_2, float lambda_1, const fpos_t *feature_start_pos, ulong_t prev_feature_id, float *beta_new, float *betaTx_delta)
{
	float beta_after_cd = 0.0;
		
	if (sum_w_x_2 != 0.0)
		beta_after_cd = soft_threshold(sum_w_q_x, lambda_1) / sum_w_x_2;
	else
		beta_after_cd = 0.0;

	fpos_t cur_fpos;
	fgetpos(file, &cur_fpos);

	fsetpos(file, feature_start_pos);
	update_betaTx(file, count, beta_after_cd - beta_new[prev_feature_id], betaTx_delta);

	fsetpos(file, &cur_fpos);

	beta_new[prev_feature_id] = beta_after_cd;
	//cout << "iter " << iter << " beta " << prev_feature_id << " " << beta_new[prev_feature_id] << "\n";

	//float qloss = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
	//printf("qloss %f\n", qloss);
}

int main(int argc, char **argv)
{
	char *dataset_filename = argv[1];
	ulong_t example_count = strtol(argv[2], NULL, 10);
	ulong_t feature_count = strtol(argv[3], NULL, 10);
	float lambda_1 = atof(argv[4]);

	float *betaTx = (float*)calloc(example_count, sizeof(float));
	float *betaTx_delta = (float*)calloc(example_count, sizeof(float));
	int *all_y = (int*)calloc(example_count, sizeof(float));
	float *beta = (float*)calloc(feature_count, sizeof(float));
	float *beta_new = (float*)calloc(feature_count, sizeof(float));

	for (int iter = 1; iter < 100; iter++) {

		FILE *file = fopen(dataset_filename, "r");
		fpos_t feature_start_pos;
		ulong_t prev_feature_id = 0;
		float loss = 0.0;

		float sum_w_x_2 = 0.0;
		float sum_w_q_x = 0.0;
		ulong_t count = 0;

		while (!feof(file)) {

			fpos_t tmp_file_pos;

			ulong_t feature_id, example_id;
			int y;
			float weight, x; 

			while (parse_line(file, &feature_id, &example_id, &y, &weight, &x, &tmp_file_pos)) {
						
				if (iter == 1)	
					all_y[example_id] = y;

				if (feature_id != prev_feature_id) {

					if (prev_feature_id != 0) {
						update_feature(file, count, sum_w_q_x, sum_w_x_2, lambda_1, &feature_start_pos, prev_feature_id, beta_new, betaTx_delta);
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
		// Linear search
		//
		float min_loss = FLT_MAX;
		float best_alpha = 0.0;

		float alpha_list[] = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};

		for (int i = 0; i < sizeof(alpha_list) / sizeof(float); i++) {
			float loss = get_loss(alpha_list[i], betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
			//printf("alpha %f loss %f\n", alpha_list[i], loss);

			if (loss < min_loss) {
				best_alpha = alpha_list[i];
				min_loss = loss;
			}
		}

		//
		// Make step
		//
		int next_newton = (iter % 5 == 0);

		if (next_newton) {
			for (int i = 0; i <= example_count; i++) {
				betaTx[i] += best_alpha * betaTx_delta[i];
				betaTx_delta[i] = 0.0;
			}

			for (int i = 0; i <= feature_count; i++) {
				beta[i] = (1 - best_alpha) * beta[i] + best_alpha * beta_new[i];
				beta_new[i] = beta[i];
			}
			cout << "NW best_alpha " << best_alpha << " min_loss " << min_loss << "\n";
		}
		else {
			cout << "CD best_alpha " << best_alpha << " min_loss " << min_loss << "\n";
		}

		//
		// Print gradient info
		//
		float *grad = get_grad(dataset_filename, example_count, feature_count, lambda_1, beta_new);

		cout << "grad ";
		for (int i = 0; i <= feature_count; i++)
			printf("%f ", grad[i]);

		free(grad);

		cout << "\n";
		cout << "\n";
 
		if (next_newton && (best_alpha == 0.0))
			break;
		
		fclose(file);
	}

	//
	// Writing features to file
	//
	FILE *file_rfeatures = fopen("rfeatures", "w");

	for (int i = 0; i < feature_count; i++) {
		fprintf(file_rfeatures, "%f\n", beta[i]);
	}

	fclose(file_rfeatures);

	free(betaTx);
	free(betaTx_delta);
	free(all_y);
}
