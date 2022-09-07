#define CL_HPP_TARGET_OPENCL_VERSION 220 
#define CL_DEVICE_TYPE_DEFAULT CL_DEVICE_TYPE_GPU 
//#define CL_HPP_NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS
#include<CL/cl.hpp>

#include<bits/stdc++.h>

struct node
{
   cl_int vertex;
   cl_int edge_weight;
};

//Reading graph edges from a file and generating adjacency list
void take_graph_input(std::string filename,
                        std::vector<std::vector<struct node>>&graph,
                        int &edge_count
                      )
{
    std::ifstream infile(filename);
    
    int v1,v2,w;
    while(infile>>v1>>v2>>w)
    {
        edge_count++;
        
        graph[v1].push_back({v2,w});
       // graph[v2].push_back({v1,w});
       
        
    }
    std::cout<<edge_count<<"\n";
}

// generate compressed sparse row representation of the graph from adjacency list
//as mentioned in the blog--> https://towardsdatascience.com/bellman-ford-single-source-shortest-path-algorithm-on-gpu-using-cuda-a358da20144b 

void generate_CSR_format(int &vertex_count,
                        std::vector<std::vector<struct node>>&graph,
                        int edge_starting_index[],
                        int edge_opposite_vertex[],
                        int dist_to[],
                        int edge_weights[])
{
    for(int i=1;i<=vertex_count;i++)
    {
        edge_starting_index[i]=edge_starting_index[i-1]+graph[i-1].size(); 
        for(int j=edge_starting_index[i],k=0;k<graph[i].size();j++,k++)
        {
            edge_opposite_vertex[j]=graph[i][k].vertex;
            edge_weights[j]=graph[i][k].edge_weight;
        }
    }
}
// void dfs(int root,
//            std::vector<std::vector<struct node>>&graph,
//             std::vector<int>&visited,int &sz)
// {
//                 std::cout<<"Entry\n";
//                 sz++;
//                 visited[root]=1;
//                 //std::cout<<"success1\n";
//                 //std::cout<<root<<",";
//                 for(auto u:graph[root])
//                 {
//                     //std::cout<<"success2\n";
                   
//                     if(!visited[u.vertex])
//                     {
//                         std::cout<<root<<"<-->"<<u.vertex<<"\n";
//                         std::cout<<"success3\n";
//                         dfs(u.vertex,graph,visited,sz);
                        
//                     }
                    
//                 }

// }

int main()
{
    std::string filename="USA-road-d.NY.gr";
    int vertex_count=264346,edge_count=0,starting_vertex=1,iscyclepresent=0,change=0;
    std::vector<std::vector<struct node>>graph(vertex_count+1); //adjacency list
   
    take_graph_input(filename,graph,edge_count);

    int edge_starting_index[vertex_count+2]={0},
    edge_opposite_vertex[edge_count+1]={0},
    dist_to[vertex_count+1]={0},
    edge_weights[edge_count+1]={0};
    
    generate_CSR_format(vertex_count,
                        graph,
                        edge_starting_index,
                        edge_opposite_vertex,
                        dist_to,
                        edge_weights);

    dist_to[starting_vertex]=0;
    edge_starting_index[vertex_count+1]=edge_count+1; //the last index of this array should point beyond the
                                                      // end of edge_opposite_vertex
    
    for(int i=1;i<=vertex_count;i++)
    {
        if(i!=starting_vertex)
        {
            dist_to[i]=INT_MAX;
        }
    }
//     std::vector<int>visited(vertex_count+1,0);
//     int size=0;
//    // visited[1]=1;
//         std::cout<<"\nroot of new component:"<<1<<",";
//             dfs(1,graph,visited,size);

   

    // std::cout<<size<<"\n"; 
    try
    {
        std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
    
            // Select the default platform and create a context using this platform and the GPU
            cl_context_properties cps[3] = { 
                CL_CONTEXT_PLATFORM, 
                (cl_context_properties)(platforms[0])(), 
                0 
            };
            cl::Context context( CL_DEVICE_TYPE_GPU, cps);
    
            // Get a list of devices on this platform
            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
            // Create a command queue and use the first device
            cl_int err;
            const cl_queue_properties _queue_properties[] =
            {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

            cl::CommandQueue queue (context, devices[0],CL_QUEUE_PROFILING_ENABLE
            );


            std::ifstream sourceFile("Bellman-Ford-adopted-version.cl");
            std::string sourceCode(
                std::istreambuf_iterator<char>(sourceFile),
                (std::istreambuf_iterator<char>()));
            cl::Program::Sources sources(1, 
                std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
    
            // Make program of the source code in the context
            cl::Program program = cl::Program(context, sources);
    
            // Build program for these specific devices
            program.build(devices);
    
            // Make kernel
            cl::Kernel kernel(program, "Bellman_Ford");

            //passing number of vertexes as a parameter to kernel
            cl::Buffer vertex_count_buffer=
                cl::Buffer(context,CL_MEM_READ_ONLY,
                sizeof(int));
            
            //this buffer[i] will represent starting index for edges associated with vertex i
            cl::Buffer edge_starting_index_buffer=
                cl::Buffer(context,CL_MEM_READ_ONLY,
                (vertex_count+1)*sizeof(int));

            
            //this buffer[i] will represent the opposite vertex of the corresponding edge associated with vertex i
            cl::Buffer  edge_opposite_vertex_buffer=
                cl::Buffer(context,CL_MEM_READ_ONLY,
                (edge_count+1)*sizeof(int));
            

            //this buffer will contain edge weights for corresponding edges
            cl::Buffer  edge_weights_buffer=
                cl::Buffer(context,CL_MEM_READ_ONLY,
                (edge_count+1)*sizeof(int));


            //this buffer will contain shortest distance to all other vertexes from source vertex
            cl::Buffer  dist_to_buffer=
                cl::Buffer(context,CL_MEM_READ_WRITE,
                (vertex_count+1)*sizeof(int));

            //this buffer acts as a flag. If for a particular iteration, none of the edges are relaxed,
            //change remains 0, so we check that and skip remaining iterations.
            cl::Buffer change_buffer=
                cl::Buffer(context,CL_MEM_READ_WRITE,
                sizeof(int));

            //Initializing the buffers/kernel parameters with appropriate values
            queue.enqueueWriteBuffer(vertex_count_buffer,
            CL_TRUE,0,sizeof(int),
            (void*)&vertex_count);

            queue.enqueueWriteBuffer(edge_starting_index_buffer,
            CL_TRUE,0,sizeof(int)*(vertex_count+1),
            (void*)edge_starting_index
            );

            queue.enqueueWriteBuffer(edge_opposite_vertex_buffer,
            CL_TRUE,0,sizeof(int)*(edge_count+1),
            (void*)edge_opposite_vertex
            );

            queue.enqueueWriteBuffer(edge_weights_buffer,
            CL_TRUE,0,sizeof(int)*(edge_count+1),
            (void*)edge_weights);

            queue.enqueueWriteBuffer(dist_to_buffer,
            CL_TRUE,0,sizeof(int)*(vertex_count+1),
            (void*)dist_to);

            

            queue.enqueueWriteBuffer(change_buffer,
            CL_TRUE,0,sizeof(int),
            (void*)&change);

            cl::Event start,stop;

            queue.enqueueMarkerWithWaitList(NULL,&start);


            //Outer loop starts, runs upto vertex_count times , that helps in negative cycle detection
            for(int i=1;i<=vertex_count;i++)
            {
                //Kernel arguments being set
                kernel.setArg(0,vertex_count_buffer);

                kernel.setArg(1,edge_starting_index_buffer);

                kernel.setArg(2,edge_opposite_vertex_buffer);

                kernel.setArg(3,edge_weights_buffer);

                kernel.setArg(4,dist_to_buffer);

               

                kernel.setArg(5,change_buffer);


                //setting total work items and number of work items in a work group.
                //This needs further study to select the optimised values for a given 
                //problem
                cl::NDRange global(vertex_count);
                cl::NDRange local(2);


                //invoking the kernel
                queue.enqueueNDRangeKernel
                (kernel, cl::NullRange,global,local);

                //read the change flag
                queue.enqueueReadBuffer(change_buffer,
                CL_TRUE,0,sizeof(int),
                (void*)&change);

                //if no change skip the remaining iterations
                if(change==0){
                    break;
                }
                else
                {
                    //if there is a negative cycle . at least one edge will be 
                    //relaxed more than vertex_count-1 times
                    if(i<vertex_count)
                    {
                        change=0;
                        queue.enqueueWriteBuffer(change_buffer,
                        CL_TRUE,0,sizeof(int),
                        (void*)&change);
                    }
                    else
                    {
                        iscyclepresent=1; //cycle detected
                    }
                }
            }
            queue.enqueueMarkerWithWaitList(NULL,&stop);

             stop.wait();

            if(!iscyclepresent)
            {
                queue.enqueueReadBuffer(dist_to_buffer,
                    CL_TRUE,0,(vertex_count+1)*(sizeof(int)),
                    (void*)&dist_to[0]
                    );
                

                for(int i=1;i<=vertex_count;i++)
                {
                    //std::cout<<dist_to[i]<<" ";
                    if(dist_to[i]!=INT_MAX)
                    std::cout<<"i:"<<i<<" dist:"<<dist_to[i]<<"\n";
                }

                //measuring and printing execution time
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
            else
            {
                std::cout<<"negative cycle present";
            }
            std::cout<<"\n";
    }
    catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        exit(0);
        }


}