#include<bits/stdc++.h>
#define INF INT_MAX
#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 220 
#define CL_DEVICE_TYPE_DEFAULT CL_DEVICE_TYPE_GPU 
#include<CL/cl.hpp>
struct node
{
    int v1,v2,edge_id;
};
bool sortbykey(const struct node &a,const struct node &b)
{
    if(a.v1!=b.v1)
    return a.v1<b.v1;
    if(a.v2!=b.v2)
    return a.v2<b.v2;
}

//reading the input bipartite graph. 
void read_graph(std::string filename,
                std::vector<struct node>&edges,
                int &edge_count,
                int &vertex_count,
                std::vector<int>&colors,
                 std::map<std::pair<int,int>,int>&edge_map)
{
    std::ifstream infile(filename);

    int u,v;
    std::map<std::pair<int,int>,int>m;
    std::map<int,int>vertex_map_X,vertex_map_Y;
    while(infile>>u>>v)
    {
        edges.push_back({u,v,1});
       
    }
    std::cout<<"reading done";
     infile.close();
    for(int i=1;i<edges.size();i++)
    {
        m[{edges[i].v1,edges[i].v2}]++;
    }
    
   edges.clear();
   edges.push_back({0,0,0});
    
    edge_count=m.size();
    std::set<int>vertex_X,vertex_Y;
    int cnt=0;
    
    for(auto u:m)
    {
        vertex_X.insert(u.first.first);
        vertex_Y.insert(u.first.second);
    }
   

    cnt=0;
    for(auto u:vertex_X)
    {
        vertex_map_X[u]=++cnt;
        colors.push_back(-1);
    }

    for(auto u:vertex_Y)
    {
        vertex_map_Y[u]=++cnt;
        colors.push_back(-1);
    }

    vertex_count=cnt;

    

    cnt=0;
    for(auto u:m)
    {
        cnt++;
        int mapped_vertex_1=vertex_map_X[u.first.first];
        int mapped_vertex_2=vertex_map_Y[u.first.second];
        edges.push_back({mapped_vertex_1,mapped_vertex_2,cnt});
        edges.push_back({mapped_vertex_2,mapped_vertex_1,cnt});

        //edge_map[{mapped_vertex_1,mapped_vertex_2}]++;
    }



    for(auto edge:edges)
    {
        colors[std::min(edge.v1,edge.v2)]=1;
       
    }


    std::cout<<"vertex count:"<<vertex_count<<"\n";
    std::sort(edges.begin(),edges.end(),sortbykey);
}

//calculating degree of each vertex , needed for CSR format
void calculate_outdegree(std::vector<int>&outdegree,
                         std::vector<struct node>&edges
                         )
{
    for(int i=1;i<edges.size();i++)
    {
        outdegree[edges[i].v1]++;
    }
    
}

// create compressed sparse row format for the given graph

void generate_csr_format(std::vector<struct node>edges,
                        std::vector<int>outdegree,
                        std::vector<int>&starting_index,
                        std::vector<int>&ending_vertex_of_edge)
{
    starting_index[1]=1;
    for(int i=1;i<edges.size();i++)
    {
        ending_vertex_of_edge[i]=edges[i].v2;

    }
    for(int i=2;i<outdegree.size();i++)
    {
        starting_index[i]=starting_index[i-1]+outdegree[i-1];
    }

}

//checking if the graph is bipartite or not
bool is_bipartite(int x,
                  int vertex_count,
                  std::vector<int>outdegree,
                  std::vector<int>starting_index,
                  std::vector<int>ending_vertex_of_edge,
                  std::vector<int>&colors)
{
    
    for(int i=1;i<=vertex_count;i++)
    {
        int current_end_index=starting_index[i]+outdegree[i]-1;
        for(int j=starting_index[i];j<=current_end_index;j++)
        {
            int edge_end_vertex=ending_vertex_of_edge[j];
            if(colors[i]==colors[edge_end_vertex])
            {
                //std::cout<<i<<" "<<edge_end_vertex<<"\n";
                return false;
            }
        }
    }
    return true;
    
}

//doing necessary initializations for opencl
void device_setup_OpenCL(std::string kernel_filename,
                         std::vector<std::string>&funcname,
                         std::vector<cl::Platform>&platforms,
                         cl::Context &context,
                         std::vector<cl::Device>&devices,
                         cl::CommandQueue &queue,
                         cl::Program &program,
                         std::vector<cl::Kernel>&kernel_arr

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
   

    for(int i=0;i<kernel_arr.size();i++)
    {
        kernel_arr[i]=cl::Kernel(program,funcname[i].c_str());
    }


    //std::cout<<"kernel set\n";
}

//setting up buffers
void buffer_setup(cl::Context context,
                  cl::CommandQueue &queue,
                  cl::Buffer buffer_arr[],
                  std::vector<struct node>&edges,
                  std::vector<int>&outdegree,
                  std::vector<int>&starting_index,
                  std::vector<int>&ending_vertex_of_edge,
                  std::vector<int>&vertices_X,
                  std::vector<int>&matched_edges,
                  std::vector<int>&parent_edge_id,
                  std::vector<int>&lookahead,
                  std::vector<int>&visited,
                  std::vector<int>&matched,
                  std::vector<int>&degree_one_vertex
                  )
                    
{
   
    buffer_arr[0]=cl::Buffer(context,CL_MEM_READ_WRITE,
                            sizeof(struct node)*(edges.size()));

    buffer_arr[1]=cl::Buffer(context,CL_MEM_READ_WRITE,
                            sizeof(int)*(outdegree.size()));

    buffer_arr[2]=cl::Buffer(context,CL_MEM_READ_WRITE,
                            sizeof(int)*(starting_index.size()));

    buffer_arr[3]=cl::Buffer(context,CL_MEM_READ_WRITE,
                            sizeof(int)*(ending_vertex_of_edge.size()));

    buffer_arr[4]=cl::Buffer(context,CL_MEM_READ_WRITE,
                            sizeof(int)*(vertices_X.size()));

    buffer_arr[5]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(int)*(matched_edges.size()));

    buffer_arr[6]=cl::Buffer(context,CL_MEM_READ_WRITE,
                             sizeof(int)*(parent_edge_id.size()));

    buffer_arr[7]=cl::Buffer(context,CL_MEM_READ_WRITE,
                                sizeof(int)*(lookahead.size()));

    buffer_arr[8]=cl::Buffer(context,CL_MEM_READ_WRITE,
                                sizeof(int)*(visited.size()));

    buffer_arr[9]=cl::Buffer(context,CL_MEM_READ_WRITE,
                                sizeof(int));

    buffer_arr[10]=cl::Buffer(context,CL_MEM_READ_WRITE,
                                sizeof(int)*(matched.size()));

    
    buffer_arr[11]=cl::Buffer(context,CL_MEM_READ_WRITE,
                            sizeof(int)*(outdegree.size()));

    

    queue.enqueueWriteBuffer(buffer_arr[0],
    CL_TRUE,0,sizeof(struct node)*(edges.size()),
    (void*)edges.data());


    
    queue.enqueueWriteBuffer(buffer_arr[1],
    CL_TRUE,0,sizeof(int)*(outdegree.size()),
    (void*)outdegree.data());


    queue.enqueueWriteBuffer(buffer_arr[2],
    CL_TRUE,0,sizeof(int)*(starting_index.size()),
    (void*)starting_index.data());

    std::cout<<"Hello\n";

    queue.enqueueWriteBuffer(buffer_arr[3],
    CL_TRUE,0,sizeof(int)*(ending_vertex_of_edge.size()),
    (void*)ending_vertex_of_edge.data());

    queue.enqueueWriteBuffer(buffer_arr[4],
    CL_TRUE,0,sizeof(int)*(vertices_X.size()),
    (void*)vertices_X.data());

   
    queue.enqueueWriteBuffer(buffer_arr[5],
    CL_TRUE,0,sizeof(int)*(matched_edges.size()),
    (void*)matched_edges.data());


    queue.enqueueWriteBuffer(buffer_arr[6],
    CL_TRUE,0,sizeof(int)*(parent_edge_id.size()),
    (void*)parent_edge_id.data());

    queue.enqueueWriteBuffer(buffer_arr[7],
    CL_TRUE,0,sizeof(int)*(lookahead.size()),
    (void*)lookahead.data());

    queue.enqueueWriteBuffer(buffer_arr[8],
    CL_TRUE,0,sizeof(int)*(visited.size()),
    (void*)visited.data());

    // queue.enqueueWriteBuffer(buffer_arr[9],
    // CL_TRUE,0,sizeof(int),
    // (void*)&path_found);

    queue.enqueueWriteBuffer(buffer_arr[10],
    CL_TRUE,0,sizeof(int)*(matched.size()),
    (void*)matched.data());

    queue.enqueueWriteBuffer(buffer_arr[11],
    CL_TRUE,0,sizeof(int)*(outdegree.size()),
    (void*)outdegree.data());

    // queue.enqueueWriteBuffer(buffer_arr[12],
    // CL_TRUE,0,sizeof(int)*(degree_one_vertex.size()),
    // (void*)degree_one_vertex.data());




      

    // queue.enqueueWriteBuffer(buffer_arr[5],
    // CL_TRUE,0,sizeof(vertex_record*)*(vertex_count+1),
    // (void*)parent_ptr);



    

}

//Parallel Karp-Sipser algorithm for initial matching

void initialize_matching(   cl::Context context,
                            cl::Buffer buffer_arr[],
                            cl::CommandQueue &queue,
                            std::vector<cl::Kernel>&kernel_arr,
                            std::vector<int>&degree_one_vertex,
                            std::vector<int>&multidegree_vertex,
                            std::vector<int>&matched_edges,
                            int vertex_count
                               )
{
    kernel_arr[0].setArg(0,buffer_arr[8]);   // __global int *visited,
        
        
    kernel_arr[0].setArg(1,buffer_arr[7]);//                         __global int *lookahead,
    kernel_arr[0].setArg(2,buffer_arr[6]);         //                         __global int *parent_edge_id,
    kernel_arr[0].setArg(3,buffer_arr[2]); //                         __global int *starting_index

    cl::NDRange global(vertex_count+1);
    cl::NDRange local(1);

    queue.enqueueNDRangeKernel
    (kernel_arr[0],cl::NullRange,global,local);

    

    // buffer_arr[12]=cl::Buffer(context,CL_MEM_READ_WRITE,
    //                             sizeof(int)*(degree_one_vertex.size()));
    // //kernel_arr[1].setArg(0,buffer_arr[12]);

    kernel_arr[1].setArg(1,buffer_arr[0]);

    kernel_arr[1].setArg(2,buffer_arr[1]);

    kernel_arr[1].setArg(3,buffer_arr[2]);

    kernel_arr[1].setArg(4,buffer_arr[3]);

    kernel_arr[1].setArg(5,buffer_arr[5]);

    kernel_arr[1].setArg(6,buffer_arr[7]);

    kernel_arr[1].setArg(7,buffer_arr[8]);

    kernel_arr[1].setArg(8,buffer_arr[6]);

    kernel_arr[1].setArg(9,buffer_arr[11]);

    kernel_arr[1].setArg(10,buffer_arr[10]);

    // if(degree_one_vertex.size()>0)
    // {
    //      buffer_arr[12]=cl::Buffer(context,CL_MEM_READ_WRITE,
    //                            sizeof(int)*(degree_one_vertex.size()));

         
    //      kernel_arr[1].setArg(0,buffer_arr[12]);

    //      global=cl::NDRange(degree_one_vertex.size());

    //     local=cl::NDRange(1);

    //     queue.enqueueNDRangeKernel
    //     (kernel_arr[1],cl::NullRange,global,local);
    // }

    

    if(multidegree_vertex.size()>0)
    {

        buffer_arr[12]=cl::Buffer(context,CL_MEM_READ_WRITE,
                                    sizeof(int)*(multidegree_vertex.size()));

        
        queue.enqueueWriteBuffer(buffer_arr[12],
        CL_TRUE,0,sizeof(int)*(multidegree_vertex.size()),
        (void*)multidegree_vertex.data());

        std::cout<<"multidegree vertices\n";

        

        kernel_arr[1].setArg(0,buffer_arr[12]);

        global=cl::NDRange(multidegree_vertex.size());

        local=cl::NDRange(1);

        queue.enqueueNDRangeKernel
        (kernel_arr[1],cl::NullRange,global,local);
    }

    queue.enqueueReadBuffer(buffer_arr[5],
    CL_TRUE,0,sizeof(int)*(matched_edges.size()),
    (void*)matched_edges.data());

    
}

//parallel Pothen-Fan algorithm
void run_pothen_fan_parallel(cl::Buffer buffer_arr[],
                                cl::CommandQueue &queue,
                                std::vector<cl::Kernel>&kernel_arr,
                                std::vector<int>vertices_X,
                                int vertex_count
                                )
{
    int path_found=0;
    do
    {
        path_found=0;
        queue.enqueueWriteBuffer(buffer_arr[9],
        CL_TRUE,0,sizeof(int),
        (void*)&path_found);

        
        kernel_arr[0].setArg(0,buffer_arr[8]);   // __global int *visited,
        
        
        kernel_arr[0].setArg(1,buffer_arr[7]);//                         __global int *lookahead,
        kernel_arr[0].setArg(2,buffer_arr[6]);         //                         __global int *parent_edge_id,
        kernel_arr[0].setArg(3,buffer_arr[2]); //                         __global int *starting_index

        cl::NDRange global(vertex_count+1);
        cl::NDRange local(1);

        queue.enqueueNDRangeKernel
        (kernel_arr[0],cl::NullRange,global,local);

        kernel_arr[2].setArg(0,buffer_arr[0]);       // __global struct node* edges,
        kernel_arr[2].setArg(1,buffer_arr[1]);       // __global int* outdegree,
        kernel_arr[2].setArg(2,buffer_arr[2]);       // __global int* starting_index,
        kernel_arr[2].setArg(3,buffer_arr[3]);      // __global int* ending_vertex_of_edge,
        kernel_arr[2].setArg(4,buffer_arr[4]);                        // __global int* vertices_X,
        kernel_arr[2].setArg(5,buffer_arr[5]);                        // __global int* matched_edges,
        kernel_arr[2].setArg(6,buffer_arr[6]);                        // __global int* parent_edge_id,
        kernel_arr[2].setArg(7,buffer_arr[7]);                        // __global int* lookahead,
        kernel_arr[2].setArg(8,buffer_arr[8]);                        // __global int* visited,


        kernel_arr[2].setArg(9,buffer_arr[9]);                        // __global int* path_found,
        kernel_arr[2].setArg(10,buffer_arr[10]);                        // __global int* matched

        global=cl::NDRange(vertices_X.size());
        local=cl::NDRange(1);

        queue.enqueueNDRangeKernel
        (kernel_arr[2],cl::NullRange,global,local);

        queue.enqueueReadBuffer(buffer_arr[9],
        CL_TRUE,0,sizeof(int),
        (void*)&path_found);
        

    } while (path_found!=0);
    
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
    std::string filename="out.discogs_style_style";
    int vertex_count=0,edge_count=0;

    std::vector<struct node>edges(1);
    std::map<std::pair<int,int>,int>edge_map;
    edges[0].v1=0,edges[0].v2=0;
    std::vector<int>colors(1);
    read_graph(filename,
                edges,
                edge_count,
                vertex_count,
                colors,
                edge_map);

    std::cout<<"graph read complete\n";

    // for(int i=1;i<=edges.size();i++)
    // {
    //     std::cout<<edges[i].v1<<" "<<edges[i].v2<<" "<<edges[i].edge_id<<"\n";
    // }
    
    std::vector<int>outdegree(vertex_count+1,0);

    

    std::vector<int>starting_index(vertex_count+1);

    std::vector<int>ending_vertex_of_edge(2*edge_count+1);

    std::vector<int>matched_edges(edge_count+1,0);

    std::vector<int>matched(vertex_count+1,-1);

    std::vector<int>parent_edge_id(vertex_count+1,-1);

    std::vector<int>lookahead(vertex_count+1,-1);

    std::vector<int>visited(vertex_count+1,0);

    std::vector<int>degree_one_vertex,multidegree_vertex;

    calculate_outdegree(outdegree,
                        edges);
    
    std::cout<<"outdegree calculated\n";
    generate_csr_format(edges,
                        outdegree,
                        starting_index,
                        ending_vertex_of_edge);

    std::cout<<"csr format generated\n";

    // for(int i=1;i<=vertex_count;i++)
    // {
    //     std::cout<<starting_index[i]<<" ";
    // }
    // std::cout<<"\n";

    // for(int i=1;i<=vertex_count;i++)
    // {
    //     std::cout<<i<<":";
    //     for(int j=starting_index[i];j<starting_index[i]+outdegree[i];j++)
    //     {
    //         std::cout<<ending_vertex_of_edge[j]<<",";
    //     }
    //     std::cout<<"\n";
    // }

   
    std::vector<int>vertices_X,vertices_Y;



    if(is_bipartite(1,
                    vertex_count,
                    outdegree,
                    starting_index,
                    ending_vertex_of_edge,
                    colors))
    {
        std::cout<<"bipartite check done\n";
        for(int i=1;i<=vertex_count;i++)
        {
            if(colors[i]==1)
            {
                vertices_X.push_back(i);
                if(outdegree[i]==1)
                {
                    degree_one_vertex.push_back(i);
                    
                }
                else if(outdegree[i]>1)
                {
                    multidegree_vertex.push_back(i);
                }
            }
           
        }

        std::cout<<"colors assigned\n";

        std::vector<cl::Platform> platforms;
            
        cl::Context context;
        std::vector<cl::Device> devices;
        cl::CommandQueue queue;
        cl::Program program;

        cl::Kernel kernel_initialize_array,kernel_match_and_update,kernel_find_augmenting_paths;

        std::vector<cl::Kernel>kernel_arr={kernel_initialize_array,
                                    kernel_match_and_update,
                                    kernel_find_augmenting_paths
                                    };


        std::string kernel_filename="Hopcroft-Karp-true-parallel.cl";
            

        std::vector<std::string>funcname={"initialize_arrays",
                                          "match_and_update",
                                          "find_augmenting_paths"
                                           
                                         };


        device_setup_OpenCL(kernel_filename,
                                funcname,
                                platforms,
                                context,
                                devices,
                                queue,
                                program,
                                kernel_arr);

         std::cout<<"Device setup complete\n";

         cl::Buffer edges_buffer,outdegree_buffer,starting_index_buffer,
         ending_vertex_of_edge_buffer,
         matched_edges_buffer,parent_edge_id_buffer,lookahead_buffer,visited_buffer,
         
         path_found_buffer,vertices_X_buffer,matched_buffer,outdegree_copy_buffer,Q_buffer;

         cl::Buffer buffer_arr[]={edges_buffer, //0
                                  outdegree_buffer,//1
                                  starting_index_buffer,//2
                                  ending_vertex_of_edge_buffer,//3
                                  vertices_X_buffer,//4
                                  matched_edges_buffer,//5
                                  parent_edge_id_buffer,//6
                                  lookahead_buffer,//7
                                  visited_buffer,//8
                                  path_found_buffer,//9
                                  matched_buffer,//10
                                  outdegree_copy_buffer,//11
                                  Q_buffer};//12};

            //  int temp_eid=starting_index[1];
            // matched_edges[edges[temp_eid].edge_id]=1;
            // matched[1]=ending_vertex_of_edge[temp_eid];
            // matched[ending_vertex_of_edge[temp_eid]]=1;
         
          buffer_setup(context,
                         queue,
                         buffer_arr,
                         edges,
                        outdegree,
                        starting_index,
                        ending_vertex_of_edge,
                        vertices_X,
                        matched_edges,
                        parent_edge_id,
                        lookahead,
                        visited,
                        matched,
                        degree_one_vertex);

            std::cout<<"Buffer setup complete\n";
           
            cl::Event start,stop;

             queue.enqueueMarkerWithWaitList(NULL,&start);

            
            initialize_matching(context,
                                buffer_arr,
                                queue,
                                kernel_arr,
                                degree_one_vertex,
                                multidegree_vertex,
                                matched_edges,
                                vertex_count);

        // for(auto u:vertices_X)
        // std::cout<<u<<" ";

            std::cout<<"Hello\n";

            std::cout<<"\n";
         
        run_pothen_fan_parallel(buffer_arr,
                                    queue,
                                    kernel_arr,
                                    vertices_X,
                                    vertex_count);

         queue.enqueueMarkerWithWaitList(NULL,&stop);
         stop.wait();

          print_execution_time(start,stop);
        // // int matching_count=0;

        queue.enqueueReadBuffer(buffer_arr[5],
        CL_TRUE,0,sizeof(int)*(edge_count+1),
        (void*)matched_edges.data());

        int matching_count=0;
        std::set<int>unique_vertices;
        std::queue<int>q;
        for(int i=1;i<=edge_count;i++)
        {
            if(matched_edges[i])
            {
                matching_count++;
                q.push(i);
            }
        }
        int cnt=0;
       
        std::cout<<"matching count:"<<matching_count<<"\n";
       
         std::cout<<"\n";

    }
    else
    {
        std::cout<<"Please provide a bipartite graph\n";
    }

  


   

    
    
    

    

    




    


}
