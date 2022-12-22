#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

bool isRangeSet(int tags){return (tags&16);}
void setTrim1(int*tags) { *tags = ( *tags | 32); };
bool isTrim1(int tags) { return ( tags & 32); }
void setTrim2(int *tags) { *tags = ( *tags | 64); };
bool isTrim2(int tags) { return ( tags & 64); };
bool isForwardVisited(int tags){ return (tags & 1);}
bool isForwardPropagate(int tags){ return (tags & 4);}
void setForwardPropagateBit(int *tags) { *tags = ( *tags | 4); };
void clearForwardPropagateBit(int *tags){*tags = ( *tags ^ 4); };
void rangeSet(int *tags) { *tags = ( *tags | 16); };
bool isBackwardVisited(int tags) { return (tags & 2); }
bool isBackwardPropagate(int tags) { return ( tags & 8); }
void setBackwardPropagateBit(int *tags) { *tags = ( *tags | 8); };
void clearBackwardPropagateBit(int *tags){ *tags = ( *tags ^ 8);};
bool isForwardProcessed(int tags){return (tags&256);};
void clearForwardProcessed(int *tags){ *tags = (*tags ^ 256);};
bool isBackwardProcessed(int tags){return (tags&512);};
void clearBackwardProcessed(int *tags){ *tags = (*tags ^ 512);};

__global int in_vertex=0,out_vertex=0;
void setPivot(int *tags) { *tags = ( *tags | 128); };
bool isPivot(int tags) { return ( tags & 128); }
//kernel_arr[0]
__kernel void trim1(__global int *range,
                  __global int *tags,
                  __global int *Fc,
                  __global int *Fr,
                  __global int *Bc,
                  __global int *Br,
                  __global unsigned *colors,
                  __global int *terminate,
                  __global int *trim1_cnt )
{

   
    int vertex=get_global_id(0);

     
    if(vertex>0)
    {
        if(isTrim1(tags[vertex]))
        return;

      //  printf("Hello,vertex=%d\n",vertex);
        
        bool eliminate=true;
        int starting_index=Br[vertex];
        int ending_index=Br[vertex+1];

      //  printf("vertex=%d,starting_index=%d,end_index=%d\n",
        //                       vertex,starting_index,ending_index);
            

        for(int i=starting_index;i<ending_index;i++)
        {
            int neighbour_vertex=Bc[i];
            
            if(!isTrim1(tags[neighbour_vertex]) && colors[vertex]==colors[neighbour_vertex])
            {
                eliminate=false;
                break;
            }
        }
       
        if(!eliminate)
        {
            eliminate=true;
            starting_index=Fr[vertex];
            ending_index=Fr[vertex+1];

            
            for(int i=starting_index;i<ending_index;i++)
            {
                int neighbour_vertex=Fc[i];
                if(neighbour_vertex<0)
                {
                    printf("This is the culprit1,vertex=%d,i=%d\n",vertex,i);
                }
                else if(neighbour_vertex>3774768)
                {
                    printf("This is the culprit2,vertex=%d,i=%d\n",vertex,i);
                }
                if(!isTrim1(tags[neighbour_vertex]) && colors[vertex]==colors[neighbour_vertex])
                {
                    eliminate=false;
                    break;
                }
            }
        }
         
        if(eliminate)
        {
            setTrim1(&tags[vertex]);
            *terminate=false;
            atomic_add(trim1_cnt,1);
           // printf("tags:%d,vertex:%d\n",tags[vertex],vertex);
        }
        
    }
  


}

//kernel_arr[1]
__kernel void pollforfirstpivot(__global int*tags,
                                __global long int*pivot_criteria,
                                __global int*Fr,
                                __global int*Br)
{
    //please set pivot_criteria to infinity in host code
     int vertex=get_global_id(0);

     if(vertex>0)
     {
        if(!isTrim1(tags[vertex]))
        {
            
            long int current_deg_product=
                     (long int)(Fr[vertex+1]-Fr[vertex])*((long int)(Br[vertex+1]-Br[vertex]));

            atom_max(pivot_criteria,current_deg_product);
            //printf("The value of deg_prod=%d\n",*pivot_criteria);
            
        }
     }   
}


//kernel_arr[2]
__kernel void selectfirstpivot(__global int*tags,
                                __global int*pivot_fields,
                                __global long int*pivot_criteria,
                                __global int*Fr,
                                __global int*Br)
{
    int vertex=get_global_id(0);

    if(vertex>0)
    {
        if(!isTrim1(tags[vertex]))
        {
            
            long int current_deg_product=
                     (long int)(Fr[vertex+1]-Fr[vertex])*((long int)(Br[vertex+1]-Br[vertex]));

            //printf("current deg product=%d,pivot criteria=%d\n",current_deg_product,*pivot_criteria);
            if(current_deg_product==(*pivot_criteria))
            {
                //atomic_xchg(&pivot_fields[0],vertex);
                if(atomic_cmpxchg(&pivot_fields[0],0,vertex)==0)
               {
                      setBackwardPropagateBit(&tags[vertex]);
                      setForwardPropagateBit(&tags[vertex]);
                      setPivot(&tags[vertex]);
                     // printf("pivot=%d\n",vertex);
                     // atomic_add(pivot_count,1);
                      

               }
                
            }

        }
    }
}
__kernel void fwd_kernel(__global int *Fr,
                         __global int *Fc,
                         __global int *tags,
                         __global unsigned *colors,
                         __global bool *terminatef
                          )
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        //printf("vertex=%d,color=%d\n",vertex,colors[vertex]);
        if(!isTrim1(tags[vertex]))
        {
            if(isForwardPropagate(tags[vertex]))
            {
                if(!isForwardProcessed(atomic_xchg(&tags[vertex],tags[vertex]|256)))
                {
                    int starting_index=Fr[vertex];
                    int ending_index=Fr[vertex+1];

                    for(int i=starting_index;i<ending_index;i++)
                    {
                        int neighbour_vertex=Fc[i];
                        if(colors[vertex]==colors[neighbour_vertex] && !isTrim1(tags[neighbour_vertex]))
                        {

                           
                                setForwardPropagateBit(&tags[neighbour_vertex]);
                                *terminatef=false;
                              
                        }                 
                    }
                }
            }
        }
    }
}

__kernel void bwd_kernel(__global int *Br,
                         __global int *Bc,
                         __global int *tags,
                         __global unsigned *colors,
                         __global bool *terminateb
                          )
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        //printf("vertex=%d,color=%d\n",vertex,colors[vertex]);
        if(!isTrim1(tags[vertex]))
        {
            if(isBackwardPropagate(tags[vertex]))
            {
                if(!isBackwardProcessed(atomic_xchg(&tags[vertex],tags[vertex]|512)))
                {
                    int starting_index=Br[vertex];
                    int ending_index=Br[vertex+1];

                    for(int i=starting_index;i<ending_index;i++)
                    {
                        int neighbour_vertex=Bc[i];
                        if(colors[vertex]==colors[neighbour_vertex] && !isTrim1(tags[neighbour_vertex]))
                        {

                            
                                setBackwardPropagateBit(&tags[neighbour_vertex]);
                                *terminateb=false;
                                //printf("vertex=%d,neighbour=%d\n",vertex,neighbour_vertex);
                              
                        }                 
                    }
                }
            }
        }
    }
}

__kernel void update(__global unsigned *colors,
                     __global int *tags,
                     __global bool *terminate
                     )
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        if(!isTrim1(tags[vertex]))
        {
            if(isForwardPropagate(tags[vertex]) && isBackwardPropagate(tags[vertex]))
            {
                setTrim1(&tags[vertex]);
            }
            else
            {
                *terminate=false;
                if(isForwardPropagate(tags[vertex])&&(!isBackwardPropagate(tags[vertex])))
                {
                    colors[vertex]=3*colors[vertex];
                    clearForwardProcessed(&tags[vertex]);
                    clearForwardPropagateBit(&tags[vertex]);
                }
                else if(isBackwardPropagate(tags[vertex])&& (!isForwardPropagate(tags[vertex])))
                {
                    colors[vertex]=3*colors[vertex]+1;
                    clearBackwardProcessed(&tags[vertex]);
                    clearBackwardPropagateBit(&tags[vertex]);
                }
                else if((!isForwardPropagate(tags[vertex])&&(!isBackwardPropagate(tags[vertex]))))
                {
                    colors[vertex]=3*colors[vertex]+2;
                }
            }
        }
    }
}

__kernel void trim2(__global int *range,
                  __global int *tags,
                  __global int *Fc,
                  __global int *Fr,
                  __global int *Bc,
                  __global int *Br,
                  __global unsigned *colors,
                  __global bool *terminate)
{
   
    int vertex=get_global_id(0);
    bool eliminate=false;
    if(vertex>0)
    {
        if(!isTrim1(vertex))
        {
            int outdegree=0;
            int sole_neighbour=-1;

            int starting_index=Fr[vertex];
            int end_index=Fr[vertex+1];

            for(int i=starting_index;i<end_index;i++)
            {
                int neighbour=Fc[i];
                if(!isTrim1(tags[neighbour]) && colors[neighbour]==colors[vertex])
                {
                    outdegree++;
                    sole_neighbour=neighbour;
              
                }       

            }
            if(outdegree==1)
            {
                outdegree=0;
                starting_index=Fr[sole_neighbour];
                end_index=Fr[sole_neighbour+1];
                int back_neighbour=-1;

                for(int i=starting_index;i<end_index;i++)
                {
                    int neighbour=Fc[i];
                    if(!isTrim1(tags[neighbour]) && colors[neighbour]==colors[vertex])
                    {
                        outdegree++;
                        back_neighbour=neighbour;
                
                    }  


                }
                if(outdegree==1 && back_neighbour==vertex)
                {
                    setTrim1(&tags[vertex]);
                    setTrim2(&tags[vertex]);
                    setTrim1(&tags[sole_neighbour]);
                    setTrim2(&tags[sole_neighbour]);
                    eliminate=true;

                }
            }

            if(!eliminate)
            {
                int indegree=0;
                starting_index=Br[vertex];
                end_index=Br[vertex+1];
                sole_neighbour=-1;

                for(int i=starting_index;i<end_index;i++)
                {
                    int neighbour=Bc[i];
                    if(!isTrim1(tags[neighbour]) && colors[neighbour]==colors[vertex])
                    {
                        indegree++;
                        sole_neighbour=neighbour;
                    }

                }
                if(indegree==1)
                {
                    indegree=0;
                    starting_index=Br[sole_neighbour];
                    end_index=Br[sole_neighbour+1];
                    int back_neighbour=-1;
                    for(int i=starting_index;i<end_index;i++)
                    {
                        int neighbour=Bc[i];
                        if(!isTrim1(tags[neighbour]) && colors[neighbour]==colors[vertex])
                        {
                            indegree++;
                            back_neighbour=neighbour;
                        }

                    }
                    if(indegree==1 && back_neighbour==vertex)
                    {
                        setTrim1(&tags[vertex]);
                        setTrim2(&tags[vertex]);
                        setTrim1(&tags[sole_neighbour]);
                        setTrim2(&tags[sole_neighbour]);
                        eliminate=true;
                    }

                }
                

            }

        }
    }
    if(eliminate)
    {
        *terminate=false;
    }
    
}

__kernel void assign_self_root(__global int *WCC,
                              __global int *tags)
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        if(!isTrim1(tags[vertex]))
        {
            WCC[vertex]=vertex;
        }
    }
}

__kernel void assign_WCC_roots(__global int *WCC,
                              __global int *tags,
                              __global unsigned *colors,
                              __global int *Fr,
                              __global int *Fc,
                              __global bool *terminate)
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        if(!isTrim1(tags[vertex]))
        {
            int starting_index=Fr[vertex];
            int ending_index=Fr[vertex+1];

            for(int i=starting_index;i<ending_index;i++)
            {
                int neighbour_vertex=Fc[i];
                if(!isTrim1(tags[neighbour_vertex]))
                {
                    if(colors[neighbour_vertex]==colors[vertex])
                    {
                        if(WCC[neighbour_vertex]<WCC[vertex])
                        {
                            WCC[vertex]=WCC[neighbour_vertex];
                            *terminate=false;
                        }
                    }
                }
            } 
        }
    }
}


__kernel void shorten_paths(__global int  *WCC,
                            __global int  *tags,
                            __global bool *terminate
                            )
{

    int vertex=get_global_id(0);
    if(vertex>0)
    {
        if(!isTrim1(tags[vertex]))
        {
            int k=WCC[vertex];

            if(k!=vertex && k!=WCC[k])
            {
                WCC[vertex]=WCC[k];
                *terminate=false;
            }
             
        }
    }
}

__kernel void color_WCC(__global int *WCC,
                        __global int *tags,
                        __global unsigned *colors)
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        
        if(!isTrim1(tags[vertex]))
        {
            // atomic_add(&in_vertex,1);
            // printf("in vertex=%d\n",in_vertex);
            int k=WCC[vertex];

            colors[vertex]=WCC[vertex];
            // atomic_add(&out_vertex,1);
            // printf("out vertex=%d\n",out_vertex);

        }
    }
}


__kernel void pollforpivots(__global unsigned *colors,
                            __global int *tags,
                            __global int *Fr,
                            __global int *Br,
                            __global long int *max_criteria,
                            __global int *vertex_count)
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        if(!isTrim1(tags[vertex])){
            unsigned c=colors[vertex];
            if(c<0)
            {
                printf("This is the culprit");
            }
            long int newdegree=((long int)(Fr[vertex+1]-Fr[vertex]))
                                *((long int)(Br[vertex+1]-Br[vertex]));
            atomic_max(&max_criteria[c%(*vertex_count)],newdegree);
            
        }
    }
}

__kernel void selectpivots(__global unsigned *colors,
                            __global int *tags,
                            __global int *Fr,
                            __global int *Br,
                            __global long int *max_criteria,
                            __global int *pivot_fields,
                            __global int *vertex_count
                            )

{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        if(!isTrim1(tags[vertex])){
            unsigned c=colors[vertex];
            long int newdegree=((long int)(Fr[vertex+1]-Fr[vertex]))
                                *((long int)(Br[vertex+1]-Br[vertex]));
            if(newdegree==max_criteria[c%(*vertex_count)])
            {
                //pivot_fields[c%(*vertex_count)]=vertex;
               if(atomic_cmpxchg(&pivot_fields[c%(*vertex_count)],0,vertex)==0)
               {
                      setBackwardPropagateBit(&tags[vertex]);
                      setForwardPropagateBit(&tags[vertex]);
                      setPivot(&tags[vertex]);
                      printf("pivot=%d\n",vertex);
                     // atomic_add(pivot_count,1);
                      

               }
            }
        }
    }
}

__kernel void remaining_vertices(__global int *tags,
                                 __global int *counter)
{
    int vertex=get_global_id(0);
    if(vertex>0)
    {
        
            if(isPivot(tags[vertex]))
            {
                   atomic_add(counter,1);
            }
        
    }
}



