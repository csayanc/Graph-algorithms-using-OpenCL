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

//function for reading the graph from file
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
        edges.push_back({v1,v2,w});                   //vector of nodes that will store input edges
        vertex_set.insert(v1),vertex_set.insert(v2); // a temporary set to get the vertex count
    }
    
    vertex_count=vertex_set.size();
    starting_vertex=*(vertex_set.begin());
    int add=0;
    if(starting_vertex==0)                          //in case the smallest vertex id starts from 0, 
                                                    // we will  start from 1.
    add=1;

    for(auto &x:edges)
    {
        x.v1=x.v1+add,x.v2=x.v2+add;
    }

}

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
    // std::cout<<"Entering device setup\n";
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

  
    queue=cl::CommandQueue(context,devices[0],CL_QUEUE_PROFILING_ENABLE);

    std::ifstream sourceFile(kernel_filename);
            std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
            cl::Program::Sources sources(1, 
                std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

   

    // Make program of the source code in the context
    program = cl::Program(context, sources);

    // Build program for these specific devices
    program.build(devices);

    kernel_find_min_edge_weight=cl::Kernel(program,kernel_funcname1.c_str());

    kernel_find_min_edge_index=cl::Kernel(program,kernel_funcname2.c_str());

    kernel_combine_components=cl::Kernel(program,kernel_funcname3.c_str());

    kernel_set_max_edge_weight=cl::Kernel(program,kernel_funcname4.c_str());


    
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

//driver function for getting component count
int count_components(cl::CommandQueue queue,
                      cl::Buffer buffer_arr[],
                      struct vertex_record* parent,
                      cl::Kernel kernel_combine_components,
                      cl::Kernel kernel_set_max_edge_weight,
                      int *current_edges,
                      int edge_count,
                      int vertex_count)
{
        
        

      
        kernel_combine_components.setArg(0,buffer_arr[0]);

        kernel_combine_components.setArg(1,buffer_arr[1]);

        kernel_combine_components.setArg(2,buffer_arr[2]);

        cl::NDRange global=cl::NDRange(edge_count);
        cl::NDRange local=cl::NDRange(1);

        queue.enqueueNDRangeKernel
        (kernel_combine_components,cl::NullRange,global,local);

        std::set<int>distinct_components;

        queue.enqueueReadBuffer(buffer_arr[2],
        CL_TRUE,0,(vertex_count+1)*sizeof(vertex_record),
        parent);                                                   //the parent array has the parent and rank information
                                                                   // for each vertex

        for(int i=1;i<=vertex_count;i++)
        {
            distinct_components.insert(get_parent(i,parent));      //distinct_components will store how many different parents
                                                                   // are there
        }
        for(auto u:distinct_components)
        std::cout<<u<<",";

        std::cout<<"\n";
        return distinct_components.size();                        //returning the total number of connected components in the graph,
                                                                  // for a connected graph this should be 1.
       //return 0;


}
int main()
{
    std::string filename="USA-road-d-for-Boruvka.txt";

    int vertex_count=0,edge_count=0,starting_vertex=-1;
    std::vector<struct node>edges(1);

    
    take_graph_input_weighted(filename,
                             edges,
                             vertex_count,
                             edge_count,
                             starting_vertex
                            );
    
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
                                                                *(vertex_count+1));             //ignore this array of pointers
        
         struct vertex_record *parent=(vertex_record*)malloc(sizeof(vertex_record) 
                                                                *(vertex_count+1));             
        
        int *min_weight=(int*)malloc(sizeof(int)*(vertex_count+1));                            //ignore for now

        int *min_weight_edge_index_component=(int*)malloc(sizeof(int)*(vertex_count+1));        //ignore for now

        for(int i=0;i<=vertex_count;i++)
        {
            
           
            parent[i].parent=i;
            parent[i].rank=0;
        }

        try
        {
            
            std::vector<cl::Platform> platforms;
            
            cl::Context context;
            std::vector<cl::Device> devices;
            cl::CommandQueue queue;
            cl::Program program;

            cl::Kernel kernel_find_min_edge_weight,kernel_find_min_edge_index,
            kernel_combine_components,                                       //this kernel has been used in the driver function 
                                                                             //above
            kernel_set_max_edge_weight;

            std::string kernel_filename="Boruvka-adopted-version-alternate.cl";
            std::string funcname1="find_min_edge_weight";
            std::string funcname2="find_min_edge_index";
            std::string funcname3="combine_components";                      //this kernel function will take an edge 
                                                                             //and perform union find between its vertices.
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

            int *current_edges=(int*)malloc(sizeof(int)*(edge_count));                       //this will store the edge index on
                                                                                             //which each thread will operate 

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

            int component_count=0;

            component_count=count_components(queue,
                                             buffer_arr,
                                             parent,
                                             kernel_combine_components,
                                             kernel_set_max_edge_weight,
                                             current_edges,
                                             edge_count,
                                             vertex_count);

            std::cout<<component_count<<"\n";

            

            

            
        }
        catch(cl::Error error)
        {
            std::cout << error.what() << "(" << error.err() << ")" << std::endl;
            exit(0);
        }
        

    }


}
