#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define INF INT_MAX

__kernel void Bellman_Ford(__global int *vertex_count,
                           __global int *edge_starting_index,
                           __global int *edge_ending_vertex,
                           __global int *edge_weight,
                            __global int *dist_to,
                            __global int *change)
{
   int i=get_global_id(0);
//    //printf("Entering\n");
//    int vc=*vertex_count;
//    //printf("vertex_count=%d\n",vc);
   if(i<((*vertex_count)+1) && i>=1)
   {
        //printf("success1\n");
        for(int j=edge_starting_index[i];j<edge_starting_index[i+1];j++)
        {
           __local int distance_to_current_vertex,distance_to_opposite_vertex;
             distance_to_current_vertex=dist_to[i];
          
             if(distance_to_current_vertex==INF)  //if the current vertex is not relaxed yet, then do not relax the edge
             {
                break;
             }
             else
             {
                distance_to_opposite_vertex=dist_to[edge_ending_vertex[j]];
                if(distance_to_current_vertex+edge_weight[j]<distance_to_opposite_vertex)  //relaxation condition
                {
                    distance_to_opposite_vertex=distance_to_current_vertex+edge_weight[j];

                    atomic_xchg(&dist_to[edge_ending_vertex[j]],
                    min(distance_to_opposite_vertex,dist_to[edge_ending_vertex[j]]));

                    *change=1;
                    //printf("success4\n");
                }
                //printf("success3\n");
             }

        }
        //printf("Hello and Welcome\n");
        
   }
}