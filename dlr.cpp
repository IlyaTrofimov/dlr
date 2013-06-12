#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<climits>
#include<iostream>

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

void update_betaTx(FILE *file, int count, float delta_beta, float *betaTx) {

	char buffer[1<<20];

	for (int i = 0; i < count; i++) {
		if (fgets(buffer, 1<<20, file)) {

			char *pstr = strtok(buffer, " ");
			ulong_t feature_id = strtoul(pstr, NULL, 10);

			pstr = strtok(NULL, " ");
			ulong_t example_id = strtoul(pstr, NULL, 10);

			pstr = strtok(NULL, " ");
			int y = strtol(pstr, NULL, 10);

			pstr = strtok(NULL, " ");
			float weight = atof(pstr);

			pstr = strtok(NULL, " ");
			float x = atof(pstr);

			betaTx[example_id] += delta_beta * x;
		}
	}
}

float get_qloss(float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y) {
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

float get_grad(float alpha, float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y) {

	float loss = 0.0;

	for (int i = 1; i <= feature_count; i++) {
		loss += 1.0 / (1.0 + exp(-betaTx[i])) * 
	}

	for (int i = 1; i <= example_count; i++) {
	
		float logit = -all_y[i] * (betaTx[i] + alpha * betaTx_delta[i]);
		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	return loss;
}
float get_loss(float alpha, float *betaTx, float *betaTx_delta, float *beta, float *beta_new, ulong_t example_count, ulong_t feature_count, float lambda_1, int *all_y) {

	float loss = 0.0;

	for (int i = 1; i <= feature_count; i++) 
		loss += lambda_1 * fabs((1 - alpha) * beta[i] + alpha * beta_new[i]);

	for (int i = 1; i <= example_count; i++) {
	
		float logit = -all_y[i] * (betaTx[i] + alpha * betaTx_delta[i]);
		if (logit > 10)
			loss += logit;
		else
			loss += log(1.0 + exp(logit));
	}

	return loss;
}

int main(int argc, char **argv)
{
	char buffer[1<<20];
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

		ulong_t feature_id = 0;

		while (!feof(file)) {

			fpos_t tmp_file_pos;
			fgetpos(file, &tmp_file_pos);

			if (fgets(buffer, 1<<20, file)) {

				char *pstr = strtok(buffer, " ");
				feature_id = strtoul(pstr, NULL, 10);

				pstr = strtok(NULL, " ");
				ulong_t example_id = strtoul(pstr, NULL, 10);

				pstr = strtok(NULL, " ");
				int y = strtol(pstr, NULL, 10);

				pstr = strtok(NULL, " ");
				float weight = atof(pstr);

				pstr = strtok(NULL, " ");
				float x = atof(pstr);
			
				if (iter == 1)	
					all_y[example_id] = y;

				float example_betaTx = betaTx[example_id];

				float p = 1.0 / (1.0 + exp(-example_betaTx));

				if (p < 1.0e-6)
					p = 1.0e-6;
				if (p > 0.999999)
					p = 0.999999;

				float w = p * (1 - p);
				float z = example_betaTx + (((y + 1) / 2) - p) / w;

				if (feature_id != prev_feature_id) {

					if (prev_feature_id != 0) {

						//float qloss1 = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
						//printf("qloss1 %f\n", qloss1);

						float new_beta = 0.0;

						if (sum_w_x_2 != 0.0)
							new_beta = soft_threshold(sum_w_q_x, lambda_1) / sum_w_x_2;
						else
							new_beta = 0.0;
					
						//printf("sum_w_q_x %f sum_w_x_2 %f\n", sum_w_q_x, sum_w_x_2);	

						// rewind to the start of current feature block, write new (beta, x) for examples
						fpos_t cur_fpos;
						fgetpos(file, &cur_fpos);

						fsetpos(file, &feature_start_pos);
						update_betaTx(file, count, new_beta - beta_new[prev_feature_id], betaTx_delta);

						fsetpos(file, &cur_fpos);

						beta_new[prev_feature_id] = new_beta;
						//cout << "iter " << iter << " beta " << prev_feature_id << " " << beta_new[prev_feature_id] << "\n";

						//float qloss = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
						//printf("qloss %f\n", qloss);

					}

			 		feature_start_pos = tmp_file_pos;

					sum_w_x_2 = 0.0;
					sum_w_q_x = 0.0;
					count = 0;
				}

				count++;

				float q = z - (example_betaTx + betaTx_delta[example_id] - beta_new[feature_id] * x);
				sum_w_x_2 += w * x * x;
				sum_w_q_x += w * q * x;
			
				prev_feature_id = feature_id;
			}
		}

		//
		float new_beta = 0.0;
		
		if (sum_w_x_2 != 0.0)
			new_beta = soft_threshold(sum_w_q_x, lambda_1) / sum_w_x_2;
		else
			new_beta = 0.0;

		fpos_t cur_fpos;
		fgetpos(file, &cur_fpos);

		fsetpos(file, &feature_start_pos);
		update_betaTx(file, count, new_beta - beta_new[prev_feature_id], betaTx_delta);

		fsetpos(file, &cur_fpos);

		beta_new[prev_feature_id] = new_beta;
		//cout << "iter " << iter << " beta " << prev_feature_id << " " << beta_new[prev_feature_id] << "\n";

		//float qloss = get_qloss(betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
		//printf("qloss %f\n", qloss);

		//
		float min_loss = 1.0e6;
		float best_alpha = 0.0;

		float alpha_list[] = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};

		for (int i = 0; i < 11; i++) {
			float loss = get_loss(alpha_list[i], betaTx, betaTx_delta, beta, beta_new, example_count, feature_count, lambda_1, all_y);
			//printf("alpha %f loss %f\n", alpha_list[i], loss);

			if (loss < min_loss) {
				best_alpha = alpha_list[i];
				min_loss = loss;
			}
		}

		int next_newton = (iter % 5 == 0);

		if (next_newton) {
			for (int i = 1; i <= example_count; i++) {
				betaTx[i] += best_alpha * betaTx_delta[i];
				betaTx_delta[i] = 0.0;
			}

			for (int i = 1; i <= feature_count; i++) {
				beta[i] = (1 - best_alpha) * beta[i] + best_alpha * beta_new[i];
				beta_new[i] = beta[i];
			}
			cout << "NW best_alpha " << best_alpha << " min_loss " << min_loss << "\n";
		}
		else {
			cout << "CD best_alpha " << best_alpha << " min_loss " << min_loss << "\n";
		}
	
		cout << "\n";
 
		if (next_newton && (best_alpha == 0.0))
			break;
		
		fclose(file);
	}

	//
	//
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
