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

void update_beta_x(FILE *file, int count, float beta_new, float beta, float lambda_1, float *betaTx) {

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

			betaTx[example_id] += (beta_new - beta) * x;
		}
	}
}

int main(int argc, char **argv)
{

	char buffer[1<<20];
	ulong_t example_count = strtol(argv[2], NULL, 10);
	float lambda_1 = atof(argv[3]);

	float *beta_x = (float*)malloc(sizeof(float) * example_count);
	float *all_y = (float*)malloc(sizeof(float) * example_count);

	float sum_w_x_2 = 0.0;
	float sum_w_q_x = 0.0;
	float beta, beta_new;
	fpos_t file_pos;
	fpos_t feature_start_pos;

	system("touch features");

	for (int iter = 1; iter < 20; iter++) {

		FILE *file = fopen(argv[1], "r");
		FILE *file_features = fopen("features", "r+");
		ulong_t prev_feature = 0;
		float loss = 0.0;
		int count;

		while (!feof(file)) {
			if (fgets(buffer, 1<<20, file)) {

				if (prev_feature == 0)
					fgetpos(file, &feature_start_pos);

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
				
				all_y[example_id] = y;
				float example_beta_x = beta_x[example_id];

				float p = 1.0 / (1.0 + exp(-example_beta_x));

				if (p < 1.0e-3)
					p = 1.0e-3;
				if (p > 0.999)
					p = 0.999;

				float w = p * (1 - p);
				float z = example_beta_x + (((y + 1) / 2) - p) / w;

				if (feature_id != prev_feature) {

					if (prev_feature != 0) {
						beta_new = soft_threshold(sum_w_q_x, lambda_1 / 2.0) / sum_w_x_2;
						if (iter > 1)
							fsetpos(file_features, &file_pos);
						fwrite((char*)&beta_new, sizeof(float), 1, file_features);
						fflush(file_features);

						fsetpos(file, &feature_start_pos);
						update_beta_x(file, count, beta_new, beta, lambda_1, beta_x);
						loss += lambda_1 * fabs(beta_new);
						fgetpos(file, &feature_start_pos);

						cout << "iter " << iter << " beta " << prev_feature << " " << beta_new << "\n";
					}

					sum_w_x_2 = 0.0;
					sum_w_q_x = 0.0;
					count = 0;

					if (iter == 1)
						beta = 0.0;
					else {
						fgetpos(file_features, &file_pos);
						fread((char*)&beta, sizeof(float), 1, file_features);
						//cout << "read beta " << beta << "\n";
					}
			
				}

				count++;

				float q = z - example_beta_x + beta * x;
				sum_w_x_2 += w * x * x;
				sum_w_q_x += w * q * x;
			
				prev_feature = feature_id;
			}
		}

		beta_new = soft_threshold(sum_w_q_x, lambda_1 / 2.0) / sum_w_x_2;
		if (iter > 1) 
			fsetpos(file_features, &file_pos);
		fwrite((char*)&beta_new, sizeof(float), 1, file_features);
		fflush(file_features);

		fsetpos(file, &feature_start_pos);
		update_beta_x(file, count, beta_new, beta, lambda_1, beta_x);
		loss += lambda_1 * fabs(beta_new);

		cout << "iter " << iter << " beta " << prev_feature << " " << beta_new << "\n";

		for (int i = 0; i < example_count; i++) {
			//printf(" %f", beta_x[i]);
			float logit = -all_y[i] * beta_x[i];
			if (logit > 10)
				loss += logit;
			else
				loss += log(1.0 + exp(logit));
		}

		cout << "loss " << loss << "\n";
		cout << "\n";

		fclose(file);
		fclose(file_features);
	}

	FILE *file_features = fopen("features", "r");
	FILE *file_rfeatures = fopen("rfeatures", "w");

	while (!feof(file_features)) {
		fread((char*)&beta, sizeof(float), 1, file_features);
		fprintf(file_rfeatures, "%f\n", beta);
	}

	fclose(file_rfeatures);
	fclose(file_features);
}
