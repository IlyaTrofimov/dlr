#include "allreduce.h"
#include "accumulate.h"
#include <cstdlib>
#include <string>
#include <cstdlib>

using std::string;

int main(int argc, char **argv) 
{
	string master_location("77.88.37.222");

	extern global_data global;

	global.unique_id = 123456;
	global.node = atoi(argv[1]);
	global.total = atoi(argv[2]);
	float value0 = atof(argv[3]);
	float value1 = atof(argv[4]);
	float value2 = atof(argv[5]);

	float v[3] = {value0, value1, value2};

	accumulate_vector(master_location, v, 3);
	printf("%f %f %f", v[0], v[1], v[2]); 

	return 1;
}
