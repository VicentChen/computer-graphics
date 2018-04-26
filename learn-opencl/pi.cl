
__kernel void sum(__global int *A,int data_size, int work_size,__global int *B)
{
	//获取local_id
	size_t local_id = get_local_id(0);
	//获取group_id
	size_t group_id = get_group_id(0);
	//获取local_size
	size_t local_size = get_local_size(0);
	//计算索引index_begin
	size_t index_begin = local_id * group_id * work_size;
	//计算索引上限
	int index_end = ((index_begin + work_size) <= data_size) ? (index_begin + work_size) : data_size;
	//求和
	int sum = 0;
	int distance2 = 0;
	for (size_t i = index_begin; i < index_end; i+=2) {
		distance2 = A[i] * A[i] + A[i+1] * A[i+1];
		if (distance2 <= 125 * 125) {
			sum++;
		}
	}
	//printf("%d %d %d %d\n", group_id, local_id,index + work_size <= data_size,sum);
	B[group_id*local_size+local_id] = sum;
}