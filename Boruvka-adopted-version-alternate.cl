#define INF INT_MAX
struct node
{
    int v1,v2,weight;
};
struct vertex_record
{
    int parent,rank;
};
//this is the find function for DSU
int get_parent(int x,struct vertex_record *parent)
{
//    //printf("x=%d\n",x);
   while (x!=parent[x].parent)
   {
     //printf("x=%d,parent=%d\n",x,parent[x].parent);
     int t=parent[x].parent;
     //printf("t=%d\n",t);
     
     //  __global
     
    __sync_bool_compare_and_swap(&parent[x].parent,t,parent[t].parent);

      x=parent[t].parent;

   }
    return x;
   
}

bool sameset(int x,int y,struct vertex_record *parent)
{
    while(true)
    {
        x=get_parent(x,parent);
       // y=get_parent(y,parent);

        if(x==y)
        return true;
        
        if(parent[x].parent==x)
        return false;
    }
}

bool updateRoot(int x,int oldrank,int y,int newrank,struct vertex_record* parent)
{
    struct vertex_record old=parent[x];
    if(old.parent!=x || old.rank!=oldrank)
    return false;

    struct vertex_record new;
    new.parent=y,new.rank=newrank;
    
    return __sync_bool_compare_and_swap((int*)&parent[x].parent,old.parent,new.parent)&& __sync_bool_compare_and_swap((int*)&parent[x].rank,old.rank,new.rank);
}

void swap(int *x,int *y)
{
    // *x=(*x+*y);
    // *y=*x-(*y);
    // *x=*x-(*y);
    int temp=*x;
    *x=*y;
    *y=temp;
}

void union_sets(int x,int y,struct vertex_record* parent)
{
    int xr,yr;
    while(true)
    {
        x=get_parent(x,parent);
        y=get_parent(y,parent);

        if(x==y)
        return;

        xr=parent[x].rank,yr=parent[y].rank;

        if(xr>yr || (xr==yr && x>y))
        {
          
             
             swap(&x,&y);
             swap(&xr,&yr);     
           


            
        }
        if(updateRoot(x,xr,y,xr,parent)==false)
        continue;
        else
        break;

    }
    if(xr==yr)
        updateRoot(y,yr,y,yr+1,parent);

   

}


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
            __global int *p=&min_weight[parent_set1];
            atomic_min(p,edges[i].weight);
            __global int *q=&min_weight[parent_set2];
            atomic_min(q,edges[i].weight);

            

        }
    }
}
__kernel void find_min_edge_index(__global struct node*edges,
                                 __global struct vertex_record *parent,
                                 __global int *min_weight,
                                 __global int *min_weight_edge_index_component)
{
    int i=get_global_id(0);
    if(i>0)
    {
        int first_vertex=edges[i].v1;
        int second_vertex=edges[i].v2;

        int parent_set1=get_parent(first_vertex,parent);
        int parent_set2=get_parent(second_vertex,parent);

        if(min_weight[parent_set1]==edges[i].weight)
        {
            min_weight_edge_index_component[parent_set1]=i;
        }
        if(min_weight[parent_set2]==edges[i].weight)
        {
            min_weight_edge_index_component[parent_set2]=i;
        }
    }
}

 


//kernel function to call union-find for each edge
__kernel void combine_components(__global int* current_edges,
                                 __global struct node* edges,
                                 __global struct vertex_record* parent

                                )
{
   
    int i=get_global_id(0);
    
    int edge_index=current_edges[i];

    int first_vertex=edges[edge_index].v1;
    int second_vertex=edges[edge_index].v2;

   

   
    union_sets(first_vertex,second_vertex,parent);
   

    
}

__kernel void set_max_edge_weight(__global int *min_weight,
                                  __global int*min_weight_edge_index_component,
                                   __global struct vertex_record* parent
                                  )
    {
        int i=get_global_id(0);
        
        min_weight[i]=INF;
        min_weight_edge_index_component[i]=-1;
    }
