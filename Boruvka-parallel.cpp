#include<bits/stdc++.h>
#define INF INT_MAX
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220 
#define CL_DEVICE_TYPE_DEFAULT CL_DEVICE_TYPE_GPU 
#include<CL/cl.hpp>

// a single node will store an edge
struct node
{
    int v1,v2,weight;
   
};
 bool compare(node &a,node &b)
    {
        if(a.v1!=b.v1)
        return a.v1<b.v1;
        else
        return a.v2<b.v2;
    }
//this struct will store parent and rank
// of the vertex which will be needed 
//during union-find    

struct vertex_record
{
    int parent,rank;
};

//thread-safe find_set function of DSU
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

bool updateRoot(int x,int oldrank,int y,int newrank,struct vertex_record* parent)
{
    struct vertex_record old=parent[x];
    if(old.parent!=x || old.rank!=oldrank)
    return false;

    struct vertex_record new_record;
    new_record.parent=y,new_record.rank=newrank;
    
    return __sync_bool_compare_and_swap((int*)&parent[x].parent,old.parent,new_record.parent)&& __sync_bool_compare_and_swap((int*)&parent[x].rank,old.rank,new_record.rank);
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
    // int xr,yr;
    // while(true)
    // {
    //     //printf("Inside union for edge %d\n",edge_index);
    //     x=get_parent(x,parent);
    //     y=get_parent(y,parent);

    //     if(x==y)
    //     return;

    //     xr=parent[x].rank,yr=parent[y].rank;

    //     if(xr>yr || (xr==yr && x>y))
    //     {
          
             
    //          swap(&x,&y);
    //          swap(&xr,&yr);     
           


            
    //     }
    //     if(updateRoot(x,xr,y,xr,parent)==false)
    //     continue;
    //     else
    //     break;

    // }
    // if(xr==yr)
    //     updateRoot(y,yr,y,yr+1,parent);
    x=get_parent(x,parent);
    y=get_parent(y,parent);

    if(x!=y)
    {
        if(parent[x].rank<parent[y].rank)
        {
            std::swap(x,y);
        }
        parent[y].parent=x;

        if(parent[x].rank==parent[y].rank)
        {
            parent[x].rank++;
        }
    }

   

}

////function for reading the graph from file
void take_graph_input_weighted(std::string filename,
                               std::vector<struct node>&edges,
                               int &vertex_count,
                               int &edge_count,
                               int &starting_vertex )
{
    std::ifstream infile(filename);
    int v1,v2,w;

    std::set<int>vertex_set;
    while(infile>>v1>>v2>>w)
    {
        edge_count++;
        edges.push_back({v1,v2,w});                     //vector of nodes that will store input edges
        vertex_set.insert(v1),vertex_set.insert(v2);    // a temporary set to get the vertex count
    }
    
    vertex_count=vertex_set.size();
    starting_vertex=*(vertex_set.begin());
    int add=0;
    if(starting_vertex==0)                              ////in case the smallest vertex id starts from 0, 
    add=1;                                               //// we will  start from 1.
                                                      
    for(auto &x:edges)
    {
        x.v1=x.v1+add,x.v2=x.v2+add;
    }
    // edges[0].v1=0,edges[1].v2=0,edges[2].weight=0;
    // std::sort(edges.begin(),edges.end(),&compare);
}
void take_graph_input_unweighted(std::string filename,
                               std::vector<struct node>&edges,
                               int &vertex_count,
                               int &edge_count,
                               int &starting_vertex )
{
    std::ifstream infile(filename);
    int v1,v2,w;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist_rand(1,100);
    

    std::set<int>vertex_set;
    while(infile>>v1>>v2)
    {
        edge_count++;
        edges.push_back({v1,v2,0});
        vertex_set.insert(v1),vertex_set.insert(v2);
    }
    
    vertex_count=vertex_set.size();
    starting_vertex=*(vertex_set.begin());
    int add=0;
    if(starting_vertex==0)
    add=1;

    for(auto &x:edges)
    {
        x.v1=x.v1+add,x.v2=x.v2+add;
        x.weight=dist_rand(rng);
    }
    // edges[0].v1=0,edges[1].v2=0,edges[2].weight=0;
    // std::sort(edges.begin(),edges.end(),&compare);
}

//
//doing the necessary initialization for kernels
void device_setup_OpenCL(std::string kernel_filename,
                         std::string kernel_funcname1,
                         std::string kernel_funcname2,
                         std::string kernel_funcname3,
                         std::string kernel_funcname4,
                         std::vector<cl::Platform>&platforms,
                         cl::Context &context,
                         std::vector<cl::Device>&devices,
                         cl::CommandQueue &queue,
                         cl::Program &program,
                         cl::Kernel &kernel_find_min_edge_weight,
                         cl::Kernel &kernel_find_min_edge_index,
                         cl::Kernel &kernel_combine_components,
                         cl::Kernel &kernel_set_max_edge_weight

)
{
    //std::cout<<"Entering device setup\n";
    cl::Platform::get(&platforms);
    
  //  std::cout<<"Platforms queried\n";
    cl_context_properties cps[3]={ 
                CL_CONTEXT_PLATFORM, 
                (cl_context_properties)(platforms[0])(), 
                0 
            };
    
  //  std::cout<<"context properties obtained\n";
    context=cl::Context(CL_DEVICE_TYPE_GPU,cps);

  //  std::cout<<"Success2\n";
    
    devices=context.getInfo<CL_CONTEXT_DEVICES>();

  //  std::cout<<"context queried\n";
    queue=cl::CommandQueue(context,devices[0],CL_QUEUE_PROFILING_ENABLE);

    std::ifstream sourceFile(kernel_filename);
            std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
            cl::Program::Sources sources(1, 
                std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

   std::cout<<"Code read complete\n";

    // Make program of the source code in the context
    program = cl::Program(context, sources);
   // std::cout<<"Hello\n";
    // Build program for these specific devices
    program.build(devices);

    std::cout<<"program built\n";
    kernel_find_min_edge_weight=cl::Kernel(program,kernel_funcname1.c_str());

    kernel_find_min_edge_index=cl::Kernel(program,kernel_funcname2.c_str());

    kernel_combine_components=cl::Kernel(program,kernel_funcname3.c_str());

    kernel_set_max_edge_weight=cl::Kernel(program,kernel_funcname4.c_str());


    //std::cout<<"kernel set\n";
}

//setting up buffers that will be passed to kernels
void buffer_setup(cl::Context context,
                  cl::CommandQueue queue,
                  cl::Buffer buffer_arr[],
                  int *current_edges,
                  std::vector<struct node>&edges,
                  struct vertex_record* parent,
                  struct vertex_record** parent_ptr,
                  int* min_weight,
                  int* min_weight_edge_index_component_buffer,
                  int vertex_count,
                  int edge_count)
                    
{
    buffer_arr[0]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(int)*(edge_count));

    buffer_arr[1]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(struct node)*(edge_count+1));
    
    buffer_arr[2]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(vertex_record)*(vertex_count+1));

    buffer_arr[3]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(int)*(vertex_count+1));

    buffer_arr[4]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(int)*(vertex_count+1));    

    buffer_arr[5]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(vertex_record*)*(vertex_count+1));    


    
    queue.enqueueWriteBuffer(buffer_arr[0],
    CL_TRUE,0,sizeof(int)*(edge_count),(void*)current_edges);

    queue.enqueueWriteBuffer(buffer_arr[1],
    CL_TRUE,0,sizeof(struct node)*(edge_count+1),(void*)edges.data());

    queue.enqueueWriteBuffer(buffer_arr[2],
    CL_TRUE,0,sizeof(vertex_record)*(vertex_count+1),(void*)parent);

    queue.enqueueWriteBuffer(buffer_arr[3],
    CL_TRUE,0,sizeof(int)*(vertex_count+1),(void*)min_weight);

    queue.enqueueWriteBuffer(buffer_arr[4],
    CL_TRUE,0,sizeof(int)*(vertex_count+1),
    (void*)min_weight_edge_index_component_buffer);

    queue.enqueueWriteBuffer(buffer_arr[5],
    CL_TRUE,0,sizeof(vertex_record*)*(vertex_count+1),
    (void*)parent_ptr);



    

}
//driver function for Boruvka MST algorithm
void runBoruvka_MST(cl::CommandQueue queue,
                    cl::Buffer buffer_arr[],
                    struct vertex_record* parent,
                    std::vector<struct node>& edges,
                    int* current_edges,
                    int vertex_count,
                    int edge_count,
                    int* min_weight_edge_index_component,
                    cl::Kernel kernel_set_max_edge_weight,
                    cl::Kernel kernel_find_min_edge_weight,
                    cl::Kernel kernel_find_min_edge_index

                    )
{
        std::set<int>mst_edges;
        int sum=0;
        
        while(mst_edges.size()<vertex_count-1)
        {
            kernel_set_max_edge_weight.setArg(0,buffer_arr[3]);

            kernel_set_max_edge_weight.setArg(1,buffer_arr[4]);

            kernel_set_max_edge_weight.setArg(2,buffer_arr[2]);

            cl::NDRange global(vertex_count+1);
            cl::NDRange local(1);

            queue.enqueueNDRangeKernel
            (kernel_set_max_edge_weight,cl::NullRange,global,local);

            kernel_find_min_edge_weight.setArg(0,buffer_arr[1]);
            kernel_find_min_edge_weight.setArg(1,buffer_arr[2]);

            kernel_find_min_edge_weight.setArg(2,buffer_arr[3]);

            global=cl::NDRange(edge_count+1);
            local=cl::NDRange(1);

            queue.enqueueNDRangeKernel
            (kernel_find_min_edge_weight,cl::NullRange,global,local);

            kernel_find_min_edge_index.setArg(0,buffer_arr[1]);
            kernel_find_min_edge_index.setArg(1,buffer_arr[2]);
            kernel_find_min_edge_index.setArg(2,buffer_arr[3]);
            kernel_find_min_edge_index.setArg(3,buffer_arr[4]);

            queue.enqueueNDRangeKernel
            (kernel_find_min_edge_index,cl::NullRange,global,local);

            queue.enqueueReadBuffer
            (buffer_arr[2],CL_TRUE,0,sizeof(vertex_record)*(vertex_count+1),
            parent);

            queue.enqueueReadBuffer
            (buffer_arr[4],CL_TRUE,0,sizeof(int)*(vertex_count+1),
            min_weight_edge_index_component);

            int flag=0;

            int mst_weight=0;

            //the edge contraction section . I have implemented it sequentially
            //as of now.
            for(int i=1;i<=vertex_count;i++)
            {
                
                if(min_weight_edge_index_component[i]!=-1)
                {
                    int edge_id=min_weight_edge_index_component[i];

                    int first_vertex=edges[edge_id].v1;

                    int second_vertex=edges[edge_id].v2;

                    int parent_set1=get_parent(first_vertex,parent);
                    int parent_set2=get_parent(second_vertex,parent);

                    if(parent_set1!=parent_set2)
                    {

                        union_sets(first_vertex,second_vertex,parent);

                        mst_edges.insert(edge_id);

                        flag=1;
                    }
                }
            }

            if(flag==0)
            {
              
               break;
            }
            queue.enqueueWriteBuffer(buffer_arr[2],
            CL_TRUE,0,sizeof(vertex_record)*(vertex_count+1),
            parent);
            
            



        } 

        std::cout<<"total edges in mst:"<<mst_edges.size()<<"\n";  
      
        long long mst_sum[vertex_count+1];
        memset(mst_sum,0,sizeof(mst_sum));
        for(auto u:mst_edges)
        {
                 int first_vertex=edges[u].v1;
                 int root=get_parent(first_vertex,parent);
                 mst_sum[root]+=edges[u].weight;

        }

        for(int i=1;i<=vertex_count;i++)
        {
            if(mst_sum[i]>0)
            {
                std::cout<<mst_sum[i]<<" ";
            }
        }

       
        

}

void print_execution_time(cl::Event start,
                           cl::Event stop )
{
    cl_ulong time_start, time_end;
    double total_time;
    start.getProfilingInfo(CL_PROFILING_COMMAND_END,
                        &time_start);
    
    stop.getProfilingInfo(CL_PROFILING_COMMAND_START,
                        &time_end);
    total_time = time_end - time_start;

    std::cout << "Execution time in milliseconds " 
    << total_time / (float)10e6<< "\n";
}

int main()
{
    std::string filename="USA-road-d-for-Boruvka.txt";

    
    int vertex_count=0,edge_count=0,starting_vertex=-1;
    std::vector<struct node>edges(1);

    char is_weighted;
    std::cout<<"Is the graph weighted? press Y for yes and N for no:";

    std::cin>>is_weighted;

    if(is_weighted=='Y')
    {
    
        take_graph_input_weighted(filename,
                                edges,
                                vertex_count,
                                edge_count,
                                starting_vertex
                                );
    }
    else
    {
        take_graph_input_unweighted(filename,
                                edges,
                                vertex_count,
                                edge_count,
                                starting_vertex
                                );
    }
    
    if(starting_vertex==-1)
    {
        std::cout<<"graph not read properly from file\n";
        exit(0);
    }
    else
    {
        std::cout<<"graph read complete\n";

        // struct vertex_record* parent=(vertex_record*)malloc
        //                                                 (sizeof(vertex_record)*(vertex_count+1));
        
      //  struct vertex_record **parent=new vertex_record*[vertex_count+1];
         struct vertex_record **parent_ptr=(vertex_record**)malloc(sizeof(vertex_record*)
                                                                *(vertex_count+1));
        
         struct vertex_record *parent=(vertex_record*)malloc(sizeof(vertex_record)
                                                                *(vertex_count+1));       //this will store parent and rank of
                                                                                          // each node
        
        int *min_weight=(int*)malloc(sizeof(int)*(vertex_count+1));                       //for storing the min weight of an edge
                                                                                          //associated with a component

        int *min_weight_edge_index_component=(int*)malloc(sizeof(int)*(vertex_count+1));  //for storing the index of the edge that
                                                                                          //has minimum weight

        for(int i=0;i<=vertex_count;i++)
        {
            
            parent_ptr[i]=NULL;
            parent[i].parent=i;
            parent[i].rank=1;
        }

        try
        {
            
            std::vector<cl::Platform> platforms;
            
            cl::Context context;
            std::vector<cl::Device> devices;
            cl::CommandQueue queue;
            cl::Program program;

            cl::Kernel kernel_find_min_edge_weight,     
            kernel_find_min_edge_index,
            kernel_combine_components,
            kernel_set_max_edge_weight;

            std::string kernel_filename="Boruvka-MST.cl";
            std::string funcname1="find_min_edge_weight";
            std::string funcname2="find_min_edge_index";
            std::string funcname3="combine_components";
            std::string funcname4="set_max_edge_weight";

            device_setup_OpenCL(kernel_filename,
                                funcname1,
                                funcname2,
                                funcname3,
                                funcname4,
                                platforms,
                                context,
                                devices,
                                queue,
                                program,
                                kernel_find_min_edge_weight,
                                kernel_find_min_edge_index,
                                kernel_combine_components,
                                kernel_set_max_edge_weight);


            std::cout<<"Device setup complete\n";
            cl::Buffer current_edges_buffer,edges_buffer,parent_buffer,
            min_weight_buffer,min_weight_edge_index_component_buffer,parent_ptr_buffer;

            cl::Buffer buffer_arr[]={ current_edges_buffer,edges_buffer,
            parent_buffer,min_weight_buffer,min_weight_edge_index_component_buffer,
            parent_ptr_buffer};

            int *current_edges=(int*)malloc(sizeof(int)*(edge_count));

            for(int i=1;i<=edge_count;i++)
            {
                current_edges[i-1]=i;
            }

            buffer_setup(context,
                         queue,
                         buffer_arr,
                         current_edges,
                         edges,
                         parent,
                         parent_ptr,
                         min_weight,
                         min_weight_edge_index_component,
                         vertex_count,
                         edge_count);

            std::cout<<"Buffer setup success\n";

            //int component_count=0;

            // component_count=count_components(queue,
            //                                  buffer_arr,
            //                                  parent,
            //                                  kernel_combine_components,
            //                                  kernel_set_max_edge_weight,
            //                                  current_edges,
            //                                  edge_count,
            //                                  vertex_count);

            cl::Event start,stop;

            queue.enqueueMarkerWithWaitList(NULL,&start);
            runBoruvka_MST(queue,
                            buffer_arr,
                            parent,
                            edges,
                            current_edges,
                            vertex_count,
                            edge_count,
                            min_weight_edge_index_component,
                            kernel_set_max_edge_weight,
                            kernel_find_min_edge_weight,
                            kernel_find_min_edge_index);
            
            queue.enqueueMarkerWithWaitList(NULL,&stop);
            
            stop.wait();

            print_execution_time(start,stop);
            
        }
        catch(cl::Error error)
        {
            std::cout << error.what() << "(" << error.err() << ")" << std::endl;
            exit(0);
        }
        

    }


}
