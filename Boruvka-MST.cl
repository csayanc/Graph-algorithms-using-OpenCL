#define INF INT_MAX
struct node
{
    int v1,v2,weight;
};
struct vertex_record
{
    int parent,rank;
};

int get_parent(int x,__global struct vertex_record *parent)
{
    
//    //printf("x=%d\n",x);
   while (x!=parent[x].parent)
   {
     //printf("Inside get parent for edge %d\n",edge_index);
     //printf("x=%d,parent=%d\n",x,parent[x].parent);
     int t=parent[x].parent;
     //printf("t=%d\n",t);
     
     //  __global
     
     __sync_bool_compare_and_swap(&parent[x].parent,t,parent[t].parent);

      x=parent[t].parent;

   }
    return x;
   
}


//finding the minimum edge weight for a particular component
__kernel void find_min_edge_weight(__global struct node *edges,
                                   __global struct vertex_record *parent,
                                   __global int *min_weight)


{
    
    int i=get_global_id(0);
    if(i>=1)
    {
        int first_vertex=edges[i].v1;
        int second_vertex=edges[i].v2;

        int parent_set1=get_parent(first_vertex,parent);
        int parent_set2=get_parent(second_vertex,parent);

        if(parent_set1!=parent_set2)
        {
            // __global int *p=&min_weight[parent_set1];
            atomic_min(&min_weight[parent_set1],edges[i].weight);
            
           //min_weight[parent_set1]=min(min_weight[parent_set1],edges[i].weight);
            
            
            // __global int *q=&min_weight[parent_set2];
            //atomic_min(q,edges[i].weight);

             atomic_min(&min_weight[parent_set2],edges[i].weight);

           // min_weight[parent_set2]=min(min_weight[parent_set2],edges[i].weight);

            

        }
    }
}

//finding the index of the corresponding edge
__kernel void find_min_edge_index(__global struct node*edges,
                                 __global struct vertex_record *parent,
                                 __global int *min_weight,
                                 __global int *min_weight_edge_index_component)
{
    
    int i=get_global_id(0);
    if(i>0)
    {
        //printf("Hello\n");
        int first_vertex=edges[i].v1;
        int second_vertex=edges[i].v2;

        int parent_set1=get_parent(first_vertex,parent);
        int parent_set2=get_parent(second_vertex,parent);

        if(parent_set1!=parent_set2)
        {

            if(min_weight[parent_set1]==edges[i].weight)
            {
                atomic_xchg(&min_weight_edge_index_component[parent_set1],i);
                
            // min_weight_edge_index_component[parent_set1]=i;
                
                
            }
            if(min_weight[parent_set2]==edges[i].weight)
            {
                atomic_xchg(&min_weight_edge_index_component[parent_set2],i);
                //min_weight_edge_index_component[parent_set2]=i;
            }
        }
    }
}

//this function should be ignored
__kernel void combine_components(__global int* current_edges,
                                 __global struct node* edges,
                                 __global struct vertex_record* parent

                                )
{
    //printf("hello\n");
    int i=get_global_id(0);
    
    int edge_index=current_edges[i];

    int first_vertex=edges[edge_index].v1;
    int second_vertex=edges[edge_index].v2;

    //printf("first vertex=%d,second_vertex=%d\n",first_vertex,second_vertex);

   
   // union_sets(first_vertex,second_vertex,parent,edge_index);
    printf("edge %d done\n",edge_index);

    
}

//this function initialises the min_weight and min_weight_edge_index_component arrays
__kernel void set_max_edge_weight(__global int *min_weight,
                                  __global int*min_weight_edge_index_component,
                                   __global struct vertex_record* parent
                                  )
    {
        int i=get_global_id(0);
        
        min_weight[i]=INF;
        min_weight_edge_index_component[i]=-1;
    }
